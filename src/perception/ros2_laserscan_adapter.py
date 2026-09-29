"""Stage 13: Real 2D LiDAR ROS 2 Interface & LaserScan Adapter.

===============================================================================
MANDATORY SENSOR & HARDWARE DISCLAIMER:
HARDWARE-READY / REAL-SENSOR-READY, BUT NOT PHYSICALLY VALIDATED
This adapter implements the ROS 2 sensor_msgs/msg/LaserScan input interface
and validation layer for physical 2D planar LiDAR integration.
Because physical 2D LiDAR sensor hardware is NOT currently available in this
development environment:
1. Physical sensor hardware has NOT been physically validated.
2. Synthetic test data is strictly segregated and must not be reported as real data.
3. The ML model, preprocessor, and schema remain frozen.
4. No vehicle actuation commands (braking, steering, throttle) are generated.
===============================================================================

Core Responsibilities:
1. Consume ROS 2 sensor_msgs/msg/LaserScan (native message, dict, or serialized payload).
2. Validate message geometry, angle resolution, range limits, beam counts, and timestamps.
3. Detect and isolate stale scans, invalid beams (NaN, Inf, negative, out-of-bounds), and corrupted geometry.
4. Maintain explicit SensorHealthState (OK, NO_DATA, STALE, INVALID, DEGRADED) decoupled from ML risk level.
5. Apply explicit vehicle coordinate frame conventions (+X forward, +Y lateral left, Z not measured)
   along with configurable mounting yaw and positional offsets.
6. Convert valid inputs into the project's LaserScan2D abstraction and forward them to the
   downstream 2D perception pipeline without duplicating extraction, tracking, or ML inference logic.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Sequence

from src.perception.lidar_2d import (
    LaserScan2D,
    LaserScanPoint2D,
    PerceptionSource,
)
from src.perception.scan_quality_2d import (
    ScanQualityAnalyzer2D,
    ScanQualityReport,
)
from src.perception.sensor_config_2d import (
    LiDAR2DConfig,
    SensorHealthState,
    SensorValidationResult,
)

logger = logging.getLogger("ROS2LaserScanAdapter")

HARDWARE_READY_DISCLAIMER: str = (
    "HARDWARE-READY / REAL-SENSOR-READY, BUT NOT PHYSICALLY VALIDATED: "
    "ROS 2 LaserScan interface implemented and software-tested. Physical 2D LiDAR "
    "hardware has not been physically validated."
)


class ROS2LaserScanAdapter(PerceptionSource):
    """Production-grade ROS 2 2D LiDAR interface adapter.

    Translates standard ROS 2 `sensor_msgs/msg/LaserScan` messages into normalized,
    coordinate-transformed `LaserScan2D` objects with strict timing and health validation.
    """

    def __init__(
        self,
        config: LiDAR2DConfig | None = None,
        time_provider: Callable[[], float] | None = None,
        is_synthetic_source: bool = False,
    ) -> None:
        """Initialize the ROS 2 LaserScan adapter.

        Args:
            config: Sensor and interface configuration parameters.
            time_provider: Optional callable returning current monotonic or epoch time in seconds
                           (useful for deterministic clock testing).
            is_synthetic_source: Flag indicating whether input data is synthetic (test harness)
                                 or real ROS 2 hardware data.
        """
        self.config = config or LiDAR2DConfig()
        self.time_provider = time_provider or time.time
        self.is_synthetic_source = is_synthetic_source

        # Operational health & telemetry state
        self._current_health: SensorHealthState = SensorHealthState.NO_DATA
        self._last_valid_scan: LaserScan2D | None = None
        self._last_scan_timestamp_ns: int = 0
        self._last_scan_received_time_s: float = 0.0
        self._total_scans_received: int = 0
        self._valid_scans_count: int = 0
        self._invalid_scans_count: int = 0
        self._stale_scans_count: int = 0
        self._last_validation_result: SensorValidationResult | None = None
        self.quality_analyzer = ScanQualityAnalyzer2D(config=self.config)
        self._last_quality_report: ScanQualityReport | None = None

    # -------------------------------------------------------------------------
    # PerceptionSource Abstract Interface Implementation
    # -------------------------------------------------------------------------

    def get_source_name(self) -> str:
        source_type = "Synthetic" if self.is_synthetic_source else "Real_ROS2_Ready"
        return f"ROS2LaserScanAdapter({source_type})"

    def is_synthetic(self) -> bool:
        return self.is_synthetic_source

    def get_sensor_dimensionality(self) -> str:
        return "2D"

    def get_latest_scan(self) -> LaserScan2D | None:
        """Retrieve the latest valid scan, enforcing stale timeout checks.

        If the last scan has become stale since reception, transitions sensor health
        to STALE and returns None (zero-actuation / zero-silent-reuse safety invariant).
        """
        if self._last_valid_scan is None:
            return None

        now_s = self.time_provider()
        scan_time_s = self._last_scan_timestamp_ns / 1e9
        age_s = now_s - scan_time_s

        if age_s > self.config.stale_timeout_s:
            if self._current_health != SensorHealthState.STALE:
                logger.warning(
                    f"Sensor scan aged out: {age_s:.3f}s exceeds stale timeout {self.config.stale_timeout_s}s. "
                    f"Transitioning sensor health to STALE."
                )
                self._current_health = SensorHealthState.STALE
            return None

        return self._last_valid_scan

    # -------------------------------------------------------------------------
    # Health & Telemetry Properties
    # -------------------------------------------------------------------------

    @property
    def current_health(self) -> SensorHealthState:
        """Current sensor health state."""
        return self._current_health

    @property
    def health_state(self) -> SensorHealthState:
        """Current sensor health state alias."""
        return self._current_health

    @property
    def last_validation_result(self) -> SensorValidationResult | None:
        return self._last_validation_result

    @property
    def total_scans_received(self) -> int:
        return self._total_scans_received

    @property
    def last_quality_report(self) -> ScanQualityReport | None:
        """Most recent scan quality diagnostic report."""
        return self._last_quality_report

    # -------------------------------------------------------------------------
    # Core Ingestion & Validation Pathway
    # -------------------------------------------------------------------------

    def ingest_laser_scan(
        self,
        msg: Any,
        current_time_s: float | None = None,
    ) -> tuple[SensorValidationResult, LaserScan2D | None]:
        """Ingest, validate, and convert a ROS 2 LaserScan message.

        Args:
            msg: ROS 2 LaserScan message, dictionary representation, or LaserScan2D object.
            current_time_s: Optional timestamp (s) to evaluate against for testing.

        Returns:
            Tuple of (SensorValidationResult, Optional[LaserScan2D]).
            If validation fails, LaserScan2D is None and SensorValidationResult details the failure.
        """
        self._total_scans_received += 1
        now_s = current_time_s if current_time_s is not None else self.time_provider()

        # 1. Null / None Check
        if msg is None:
            self._invalid_scans_count += 1
            if self._total_scans_received == 1:
                self._current_health = SensorHealthState.NO_DATA
            else:
                self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=self._current_health,
                message="LaserScan message is None",
            )
            self._last_validation_result = res
            return res, None

        # 2. Extract Fields (supports ROS 2 msg object, dict, or LaserScan2D)
        try:
            extracted = self._extract_raw_fields(msg)
        except Exception as e:
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message=f"Failed to parse LaserScan structure: {e}",
            )
            self._last_validation_result = res
            return res, None

        # 3. Geometric & Range Bounds Validation
        ranges = extracted["ranges"]
        angle_min = extracted["angle_min"]
        angle_max = extracted["angle_max"]
        angle_inc = extracted["angle_increment"]
        range_min = extracted["range_min"]
        range_max = extracted["range_max"]
        timestamp_ns = extracted["timestamp_ns"]
        frame_id = extracted["frame_id"]

        if not frame_id or not str(frame_id).strip():
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message="LaserScan frame_id is missing or empty",
            )
            self._last_validation_result = res
            return res, None

        if self.config.enforce_frame_id and str(frame_id) != self.config.expected_frame_id:
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message=f"Frame ID mismatch: expected '{self.config.expected_frame_id}', got '{frame_id}'",
            )
            self._last_validation_result = res
            return res, None

        if not ranges or len(ranges) == 0:
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message="LaserScan ranges array is empty",
            )
            self._last_validation_result = res
            return res, None

        if angle_inc <= 0.0 or math.isnan(angle_inc) or math.isinf(angle_inc):
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message=f"Invalid angle_increment: {angle_inc} (must be > 0)",
            )
            self._last_validation_result = res
            return res, None

        if angle_min >= angle_max or math.isnan(angle_min) or math.isnan(angle_max):
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message=f"Invalid angle bounds: angle_min ({angle_min}) >= angle_max ({angle_max})",
            )
            self._last_validation_result = res
            return res, None

        if range_min < 0.0 or range_max <= 0.0 or range_min >= range_max:
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message=f"Invalid range limits: range_min={range_min}, range_max={range_max}",
            )
            self._last_validation_result = res
            return res, None

        # Expected beam count sanity check
        expected_beams = int(round((angle_max - angle_min) / angle_inc)) + 1
        actual_beams = len(ranges)
        # Allow small rounding tolerance (+/- 2 beams)
        if abs(actual_beams - expected_beams) > max(2, int(0.05 * expected_beams)):
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message=f"Beam count mismatch: received {actual_beams}, expected approximately {expected_beams}",
                total_beam_count=actual_beams,
            )
            self._last_validation_result = res
            return res, None

        # 4. Timestamp & Stale Data Check
        scan_time_s = timestamp_ns / 1e9 if timestamp_ns > 0 else now_s
        age_s = now_s - scan_time_s
        timestamp_gap_s = 0.0

        if self._last_scan_timestamp_ns > 0:
            timestamp_gap_s = (timestamp_ns - self._last_scan_timestamp_ns) / 1e9

        if age_s > self.config.stale_timeout_s:
            self._stale_scans_count += 1
            self._current_health = SensorHealthState.STALE
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.STALE,
                message=f"Stale scan detected: age {age_s:.3f}s exceeds timeout {self.config.stale_timeout_s}s",
                age_s=age_s,
                timestamp_gap_s=timestamp_gap_s,
                total_beam_count=actual_beams,
            )
            self._last_validation_result = res
            return res, None

        if self.config.max_future_skew_s is not None and age_s < -self.config.max_future_skew_s:
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message=f"Future timestamp detected: scan timestamp is {-age_s:.3f}s in the future",
                age_s=age_s,
                timestamp_gap_s=timestamp_gap_s,
                total_beam_count=actual_beams,
            )
            self._last_validation_result = res
            return res, None

        # 5. Per-Beam Signal Quality Analysis
        valid_return_count = 0
        free_space_count = 0
        corrupted_count = 0

        clean_ranges: list[float] = []
        for r in ranges:
            if r is None or math.isnan(r):
                clean_ranges.append(float("nan"))
                corrupted_count += 1
            elif math.isinf(r):
                if r > 0:
                    clean_ranges.append(float("inf"))
                    free_space_count += 1
                else:
                    clean_ranges.append(float("-inf"))
                    corrupted_count += 1
            elif r < 0.0 or (0.0 < r < range_min):
                clean_ranges.append(float(r))
                corrupted_count += 1
            elif r > range_max:
                clean_ranges.append(float(r))
                free_space_count += 1
            else:
                clean_ranges.append(float(r))
                valid_return_count += 1

        total_beams = len(clean_ranges)
        # If all beams in the scan are corrupted (all NaN, negative, or -inf)
        if corrupted_count == total_beams:
            self._invalid_scans_count += 1
            self._current_health = SensorHealthState.INVALID
            res = SensorValidationResult(
                is_valid=False,
                health_state=SensorHealthState.INVALID,
                message="All beams in scan are corrupted, NaN, or negative",
                valid_beam_count=0,
                total_beam_count=total_beams,
                valid_beam_ratio=0.0,
                age_s=age_s,
                timestamp_gap_s=timestamp_gap_s,
            )
            self._last_validation_result = res
            return res, None

        # Check if sensor stream is degraded due to high noise / occlusion
        corrupted_ratio = corrupted_count / total_beams
        if corrupted_ratio > (1.0 - self.config.min_valid_beam_ratio):
            self._current_health = SensorHealthState.DEGRADED
            health_state = SensorHealthState.DEGRADED
            status_msg = f"Degraded sensor health: {corrupted_count}/{total_beams} corrupted beams ({corrupted_ratio * 100:.1f}%)"
        else:
            self._current_health = SensorHealthState.OK
            health_state = SensorHealthState.OK
            status_msg = "Scan valid and within nominal parameters"

        # 6. Construct Coordinate-Transformed LaserScan2D
        scan2d = LaserScan2D(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            angle_min=angle_min,
            angle_max=angle_max,
            angle_increment=angle_inc,
            time_increment=extracted["time_increment"],
            scan_time=extracted["scan_time"],
            range_min=range_min,
            range_max=range_max,
            ranges=clean_ranges,
            intensities=extracted["intensities"],
            mounting_yaw_offset_rad=self.config.mounting_yaw_offset_rad,
            mounting_x_offset_m=self.config.mounting_x_offset_m,
            mounting_y_offset_m=self.config.mounting_y_offset_m,
        )

        self._valid_scans_count += 1
        self._last_valid_scan = scan2d
        self._last_scan_timestamp_ns = timestamp_ns
        self._last_scan_received_time_s = now_s

        quality_report = self.quality_analyzer.analyze_scan(scan2d, current_time_s=now_s)
        self._last_quality_report = quality_report

        res = SensorValidationResult(
            is_valid=True,
            health_state=health_state,
            message=status_msg,
            valid_beam_count=valid_return_count,
            total_beam_count=total_beams,
            valid_beam_ratio=valid_return_count / total_beams if total_beams > 0 else 0.0,
            age_s=age_s,
            timestamp_gap_s=timestamp_gap_s,
            quality_report=quality_report.to_dict(),
        )
        self._last_validation_result = res
        return res, scan2d

    # -------------------------------------------------------------------------
    # Helper: Extract Raw Fields from Heterogeneous Input Formats
    # -------------------------------------------------------------------------

    def _extract_raw_fields(self, msg: Any) -> dict[str, Any]:
        """Extract standard LaserScan fields from ROS 2 message, dict, or LaserScan2D."""
        if isinstance(msg, LaserScan2D):
            return {
                "frame_id": msg.frame_id,
                "timestamp_ns": msg.timestamp_ns,
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "time_increment": msg.time_increment,
                "scan_time": msg.scan_time,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
                "ranges": list(msg.ranges),
                "intensities": list(msg.intensities),
            }

        if isinstance(msg, dict):
            header = msg.get("header", {})
            stamp = header.get("stamp", {}) if isinstance(header, dict) else getattr(header, "stamp", {})
            sec = stamp.get("sec", 0) if isinstance(stamp, dict) else getattr(stamp, "sec", 0)
            nanosec = stamp.get("nanosec", 0) if isinstance(stamp, dict) else getattr(stamp, "nanosec", 0)
            timestamp_ns = (sec * 1_000_000_000) + nanosec
            if timestamp_ns == 0:
                timestamp_ns = int(msg.get("timestamp_ns", 0))

            if isinstance(header, dict) and "frame_id" in header:
                frame_id = header["frame_id"]
            elif hasattr(header, "frame_id"):
                frame_id = getattr(header, "frame_id")
            elif "frame_id" in msg:
                frame_id = msg["frame_id"]
            else:
                frame_id = "laser_frame"

            return {
                "frame_id": str(frame_id) if frame_id is not None else "",
                "timestamp_ns": timestamp_ns,
                "angle_min": float(msg["angle_min"]),
                "angle_max": float(msg["angle_max"]),
                "angle_increment": float(msg["angle_increment"]),
                "time_increment": float(msg.get("time_increment", 0.0)),
                "scan_time": float(msg.get("scan_time", 0.05)),
                "range_min": float(msg["range_min"]),
                "range_max": float(msg["range_max"]),
                "ranges": list(msg.get("ranges", [])),
                "intensities": list(msg.get("intensities", [])),
            }

        # Handle native ROS 2 sensor_msgs/msg/LaserScan or duck-typed object
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None) if header else None
        sec = getattr(stamp, "sec", 0) if stamp else 0
        nanosec = getattr(stamp, "nanosec", 0) if stamp else 0
        timestamp_ns = (sec * 1_000_000_000) + nanosec
        if header and hasattr(header, "frame_id"):
            frame_id = getattr(header, "frame_id")
        elif hasattr(msg, "frame_id"):
            frame_id = getattr(msg, "frame_id")
        else:
            frame_id = "laser_frame"

        return {
            "frame_id": str(frame_id),
            "timestamp_ns": timestamp_ns,
            "angle_min": float(getattr(msg, "angle_min")),
            "angle_max": float(getattr(msg, "angle_max")),
            "angle_increment": float(getattr(msg, "angle_increment")),
            "time_increment": float(getattr(msg, "time_increment", 0.0)),
            "scan_time": float(getattr(msg, "scan_time", 0.05)),
            "range_min": float(getattr(msg, "range_min")),
            "range_max": float(getattr(msg, "range_max")),
            "ranges": list(getattr(msg, "ranges", [])),
            "intensities": list(getattr(msg, "intensities", [])),
        }
