"""Stage 14: Offline 2D LiDAR Calibration & Sensor Validation Tool.

===============================================================================
MANDATORY HARDWARE DISCLAIMER:
OFFLINE VALIDATION TOOLING — SYNTHETIC / REPLAY VALIDATION
Physical 2D LiDAR hardware is not connected in this development environment.
All outputs from synthetic or replay evaluations are explicitly marked as:
HARDWARE STATUS: NOT VALIDATED — HARDWARE NOT AVAILABLE
===============================================================================

Provides comprehensive offline analysis of recorded (or simulated) 2D LaserScan
datasets, evaluating:
1. Sensor configuration & calibration integrity
2. Scan quality metrics (beam counts, valid %, NaN %, Inf %, coverage, frequency)
3. Sensor health state transitions (OK, NO_DATA, STALE, INVALID, DEGRADED)
4. Obstacle clustering & spatial distribution
5. Temporal multi-scan tracking & closing velocity statistics
6. 2D-to-model feature compatibility audit
7. Frozen ML risk prediction distribution
8. Prediction provenance metadata
9. Software processing latency breakdown
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.feature_mapper_2d import FeatureMapper2D
from src.perception.lidar_2d import LaserScan2D
from src.perception.lidar_calibration_2d import LiDARCalibration2D
from src.perception.lidar_scan_replay import LiDARScanReplayer
from src.perception.obstacle_extractor_2d import (
    MultiScanTracker2D,
    ObstacleExtractor2D,
)
from src.perception.ros2_laserscan_adapter import ROS2LaserScanAdapter
from src.perception.scan_quality_2d import ScanQualityAnalyzer2D, ScanQualityReport
from src.perception.sensor_config_2d import (
    LiDAR2DConfig,
    SensorHealthState,
    SensorValidationResult,
)
from minerakshak_risk.risk_node import PerceptionToRiskProcessor
from minerakshak_risk.schemas import RiskAssessmentFrame

OFFLINE_TOOL_DISCLAIMER = (
    "OFFLINE VALIDATION TOOLING — SYNTHETIC / REPLAY VALIDATION: "
    "Physical 2D LiDAR hardware is not connected. Results are from replay/synthetic "
    "data and must not be reported as physical hardware validation."
)


@dataclass
class ValidationSessionSummary:
    """Aggregated results from an offline 2D LiDAR validation run."""

    tool_disclaimer: str = OFFLINE_TOOL_DISCLAIMER
    hardware_status: str = "NOT VALIDATED — HARDWARE NOT AVAILABLE"
    data_source: str = "synthetic_replay"
    total_frames_processed: int = 0
    total_processing_wall_time_s: float = 0.0

    # Sensor Health State Distribution
    health_state_counts: dict[str, int] = field(default_factory=lambda: {
        "OK": 0, "DEGRADED": 0, "STALE": 0, "INVALID": 0, "NO_DATA": 0,
    })

    # Scan Quality Metrics Summary
    mean_valid_beam_percentage: float = 0.0
    mean_nan_beam_percentage: float = 0.0
    mean_infinite_beam_percentage: float = 0.0
    mean_scan_coverage_deg: float = 0.0
    min_observed_range_m: float | None = None
    max_observed_range_m: float | None = None

    # Obstacle & Tracking Statistics
    total_obstacles_detected: int = 0
    mean_obstacles_per_frame: float = 0.0
    total_tracked_obstacles: int = 0
    mean_relative_velocity_mps: float = 0.0
    min_time_to_collision_s: float | None = None

    # Risk Prediction Distribution
    risk_tier_counts: dict[str, int] = field(default_factory=lambda: {
        "SAFE": 0, "CAUTION": 0, "WARNING": 0, "CRITICAL": 0,
    })

    # Feature Compatibility Summary
    nominal_defaults_used: dict[str, int] = field(default_factory=lambda: {
        "object_z_m_default": 0,
        "object_height_m_default": 0,
        "object_type_default": 0,
    })

    # Latency Breakdown (milliseconds)
    latency_ms: dict[str, float] = field(default_factory=lambda: {
        "validation_mean_ms": 0.0,
        "clustering_mean_ms": 0.0,
        "tracking_mean_ms": 0.0,
        "inference_mean_ms": 0.0,
        "total_cycle_mean_ms": 0.0,
        "total_cycle_p95_ms": 0.0,
        "total_cycle_p99_ms": 0.0,
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OfflineValidator2D:
    """Offline calibration, quality, and pipeline validator for 2D LaserScan streams."""

    def __init__(
        self,
        config: LiDAR2DConfig | None = None,
        truck_speed_kmph: float = 25.0,
    ) -> None:
        self.config = config or LiDAR2DConfig()
        self.truck_speed_kmph = truck_speed_kmph
        self.calibrator = LiDARCalibration2D(config=self.config)
        self.quality_analyzer = ScanQualityAnalyzer2D(config=self.config)
        self.adapter = ROS2LaserScanAdapter(config=self.config, is_synthetic_source=True)
        self.extractor = ObstacleExtractor2D(cluster_distance_threshold_m=1.2, min_cluster_points=2)
        self.tracker = MultiScanTracker2D(gating_distance_m=4.5, velocity_smoothing_alpha=0.7)
        self.mapper = FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5)
        self.risk_processor = PerceptionToRiskProcessor()

    def run_validation_on_records(
        self,
        scan_records: Sequence[dict[str, Any]],
        source_label: str = "synthetic_replay",
    ) -> ValidationSessionSummary:
        """Run complete validation pipeline over a sequence of LaserScan dictionaries."""
        summary = ValidationSessionSummary(data_source=source_label)
        t_session_start = time.perf_counter()

        if not scan_records:
            summary.total_processing_wall_time_s = time.perf_counter() - t_session_start
            return summary

        total_frames = len(scan_records)
        summary.total_frames_processed = total_frames

        valid_beam_pcts: list[float] = []
        nan_beam_pcts: list[float] = []
        inf_beam_pcts: list[float] = []
        coverages_deg: list[float] = []
        min_ranges: list[float] = []
        max_ranges: list[float] = []
        velocities: list[float] = []
        ttcs: list[float] = []

        val_latencies: list[float] = []
        clust_latencies: list[float] = []
        track_latencies: list[float] = []
        inf_latencies: list[float] = []
        total_latencies: list[float] = []

        for record in scan_records:
            t_frame_start = time.perf_counter()

            # 1. Validation & Conversion
            t_v0 = time.perf_counter()
            val_res, scan2d = self.adapter.ingest_laser_scan(record)
            val_lat = (time.perf_counter() - t_v0) * 1000.0
            val_latencies.append(val_lat)

            health_state_str = val_res.health_state.value
            summary.health_state_counts[health_state_str] = (
                summary.health_state_counts.get(health_state_str, 0) + 1
            )

            if not val_res.is_valid or scan2d is None:
                total_lat = (time.perf_counter() - t_frame_start) * 1000.0
                total_latencies.append(total_lat)
                continue

            # Quality metrics
            q_rep = self.quality_analyzer.analyze_scan(scan2d)
            valid_beam_pcts.append(q_rep.valid_beam_percentage)
            nan_beam_pcts.append(q_rep.nan_beam_percentage)
            inf_beam_pcts.append(q_rep.infinite_beam_percentage)
            coverages_deg.append(q_rep.scan_coverage_deg)
            if q_rep.min_valid_range_m is not None:
                min_ranges.append(q_rep.min_valid_range_m)
            if q_rep.max_valid_range_m is not None:
                max_ranges.append(q_rep.max_valid_range_m)

            # 2. Obstacle Extraction
            t_c0 = time.perf_counter()
            obstacles = self.extractor.extract_obstacles(scan2d)
            clust_lat = (time.perf_counter() - t_c0) * 1000.0
            clust_latencies.append(clust_lat)
            summary.total_obstacles_detected += len(obstacles)

            # 3. Multi-Scan Tracking
            t_t0 = time.perf_counter()
            tracked_obstacles = self.tracker.track(obstacles, timestamp_ns=scan2d.timestamp_ns)
            track_lat = (time.perf_counter() - t_t0) * 1000.0
            track_latencies.append(track_lat)
            summary.total_tracked_obstacles += len(tracked_obstacles)

            for obs in tracked_obstacles:
                velocities.append(obs.relative_velocity_mps)
                if obs.time_to_collision_s < 999.0:
                    ttcs.append(obs.time_to_collision_s)

            # 4. Feature Mapping
            perception_frame = self.mapper.to_perception_frame(
                obstacles=tracked_obstacles,
                truck_speed_kmph=self.truck_speed_kmph,
                frame_id=scan2d.frame_id,
                timestamp_ns=scan2d.timestamp_ns,
                sensor_source=self.adapter.get_source_name(),
                source_type=source_label,
                hardware_validated=False,
                sensor_health=val_res.health_state.value,
                feature_compatibility_status="COMPATIBLE_WITH_DEFAULTS",
            )
            # Track nominal compatibility defaults used
            summary.nominal_defaults_used["object_z_m_default"] += len(tracked_obstacles)
            summary.nominal_defaults_used["object_height_m_default"] += len(tracked_obstacles)
            summary.nominal_defaults_used["object_type_default"] += len(tracked_obstacles)

            # 5. ML Risk Inference
            t_i0 = time.perf_counter()
            risk_frame = self.risk_processor.process_frame(perception_frame)
            inf_lat = (time.perf_counter() - t_i0) * 1000.0
            inf_latencies.append(inf_lat)

            # Accumulate risk predictions
            for assessment in risk_frame.assessments:
                tier = assessment.predicted_risk_level
                summary.risk_tier_counts[tier] = summary.risk_tier_counts.get(tier, 0) + 1

            total_lat = (time.perf_counter() - t_frame_start) * 1000.0
            total_latencies.append(total_lat)

        # Compute summary aggregations
        summary.total_processing_wall_time_s = time.perf_counter() - t_session_start
        if total_frames > 0:
            summary.mean_obstacles_per_frame = round(summary.total_obstacles_detected / total_frames, 2)

        if valid_beam_pcts:
            summary.mean_valid_beam_percentage = round(sum(valid_beam_pcts) / len(valid_beam_pcts), 2)
            summary.mean_nan_beam_percentage = round(sum(nan_beam_pcts) / len(nan_beam_pcts), 2)
            summary.mean_infinite_beam_percentage = round(sum(inf_beam_pcts) / len(inf_beam_pcts), 2)
            summary.mean_scan_coverage_deg = round(sum(coverages_deg) / len(coverages_deg), 2)

        if min_ranges:
            summary.min_observed_range_m = round(min(min_ranges), 3)
        if max_ranges:
            summary.max_observed_range_m = round(max(max_ranges), 3)

        if velocities:
            summary.mean_relative_velocity_mps = round(sum(velocities) / len(velocities), 3)
        if ttcs:
            summary.min_time_to_collision_s = round(min(ttcs), 2)

        # Latency statistics
        if total_latencies:
            sorted_totals = sorted(total_latencies)
            n = len(sorted_totals)
            p95_idx = min(int(round(0.95 * n)), n - 1)
            p99_idx = min(int(round(0.99 * n)), n - 1)

            summary.latency_ms = {
                "validation_mean_ms": round(sum(val_latencies) / len(val_latencies), 3) if val_latencies else 0.0,
                "clustering_mean_ms": round(sum(clust_latencies) / len(clust_latencies), 3) if clust_latencies else 0.0,
                "tracking_mean_ms": round(sum(track_latencies) / len(track_latencies), 3) if track_latencies else 0.0,
                "inference_mean_ms": round(sum(inf_latencies) / len(inf_latencies), 3) if inf_latencies else 0.0,
                "total_cycle_mean_ms": round(sum(total_latencies) / n, 3),
                "total_cycle_p95_ms": round(sorted_totals[p95_idx], 3),
                "total_cycle_p99_ms": round(sorted_totals[p99_idx], 3),
            }

        return summary

    def run_on_replay_file(self, jsonl_path: Path | str) -> ValidationSessionSummary:
        """Run validation on a recorded JSONL scan dataset."""
        replayer = LiDARScanReplayer(jsonl_path)
        records = replayer.get_all_scans()
        source_label = f"recorded_replay({Path(jsonl_path).name})"
        return self.run_validation_on_records(records, source_label=source_label)


def main() -> None:
    """CLI entry point for offline 2D LiDAR validation tooling."""
    parser = argparse.ArgumentParser(
        description="MineRakshak AI — Stage 14 Offline 2D LiDAR Calibration & Validation Tool",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="",
        help="Path to recorded LaserScan dataset (.jsonl). If omitted, a synthetic test stream is used.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Path to write JSON validation report.",
    )
    parser.add_argument(
        "--truck-speed",
        type=float,
        default=25.0,
        help="Vehicle speed in km/h (default: 25.0).",
    )
    args = parser.parse_args()

    validator = OfflineValidator2D(truck_speed_kmph=args.truck_speed)

    if args.input and Path(args.input).exists():
        print(f"[STAGE 14 OFFLINE VALIDATOR] Ingesting recorded scans: {args.input}")
        summary = validator.run_on_replay_file(args.input)
    else:
        print("[STAGE 14 OFFLINE VALIDATOR] Generating synthetic 2D scan sequence for validation...")
        # Generate 20 synthetic test scans
        import time as pytime
        now_ns = int(pytime.time() * 1e9)
        dt_ns = 50_000_000  # 20 Hz
        synthetic_records: list[dict[str, Any]] = []

        # Create moving obstacle: begins at 35m, closes at -5 m/s
        for step in range(25):
            t_ns = now_ns + (step * dt_ns)
            dist = max(5.0, 35.0 - (5.0 * step * 0.05))
            ranges = [float("inf")] * 361
            # Place obstacle around forward direction (index 175 to 185)
            for b in range(175, 186):
                ranges[b] = dist + ((b - 180) * 0.02)

            synthetic_records.append({
                "header": {"frame_id": "laser_frame", "stamp": {"sec": t_ns // 1_000_000_000, "nanosec": t_ns % 1_000_000_000}},
                "angle_min": -math.pi / 2,
                "angle_max": math.pi / 2,
                "angle_increment": math.pi / 360,
                "time_increment": 0.0,
                "scan_time": 0.05,
                "range_min": 0.1,
                "range_max": 80.0,
                "ranges": ranges,
                "intensities": [100.0] * 361,
            })

        summary = validator.run_validation_on_records(synthetic_records, source_label="synthetic_generated_sequence")

    print("\n" + "=" * 70)
    print("STAGE 14 OFFLINE 2D LIDAR VALIDATION REPORT")
    print("=" * 70)
    print(f"DISCLAIMER:           {summary.tool_disclaimer}")
    print(f"HARDWARE STATUS:      {summary.hardware_status}")
    print(f"DATA SOURCE:          {summary.data_source}")
    print(f"FRAMES PROCESSED:     {summary.total_frames_processed}")
    print(f"WALL CLOCK TIME:      {summary.total_processing_wall_time_s * 1000:.2f} ms")
    print("\n--- SENSOR HEALTH STATE TRANSITIONS ---")
    for k, v in summary.health_state_counts.items():
        print(f"  {k:12s}: {v}")
    print("\n--- SCAN QUALITY SUMMARY ---")
    print(f"  Mean Valid Beams:   {summary.mean_valid_beam_percentage:.1f}%")
    print(f"  Mean NaN Beams:     {summary.mean_nan_beam_percentage:.1f}%")
    print(f"  Mean Inf Beams:     {summary.mean_infinite_beam_percentage:.1f}%")
    print(f"  Mean Coverage:      {summary.mean_scan_coverage_deg:.1f} deg")
    print(f"  Range Bounds:       [{summary.min_observed_range_m} m, {summary.max_observed_range_m} m]")
    print("\n--- OBSTACLE & TRACKING STATISTICS ---")
    print(f"  Obstacles Detected: {summary.total_obstacles_detected} (Mean: {summary.mean_obstacles_per_frame}/frame)")
    print(f"  Tracked Obstacles:  {summary.total_tracked_obstacles}")
    print(f"  Mean Rel. Velocity: {summary.mean_relative_velocity_mps:.2f} m/s")
    print(f"  Min Observed TTC:   {summary.min_time_to_collision_s} s")
    print("\n--- RISK PREDICTIONS ---")
    for tier, count in summary.risk_tier_counts.items():
        print(f"  {tier:10s}: {count}")
    print("\n--- LATENCY BREAKDOWN (SOFTWARE BENCHMARK) ---")
    for k, v in summary.latency_ms.items():
        print(f"  {k:22s}: {v:.3f} ms")
    print("=" * 70)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"[OK] Report written to: {args.output}")


if __name__ == "__main__":
    main()
