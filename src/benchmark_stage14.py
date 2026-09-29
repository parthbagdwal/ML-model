"""Stage 14: 2D LiDAR Calibration, Perception & Inference Latency Benchmark.

===============================================================================
MANDATORY BENCHMARK DISCLAIMER:
SOFTWARE BENCHMARK — NOT PHYSICAL SENSOR LATENCY
This benchmark measures software processing execution time on the host CPU.
It does NOT measure physical 2D LiDAR hardware bus latency, laser pulse time-of-flight,
or serial/Ethernet transmission delay. Physical hardware is not connected.
===============================================================================

Profiles every sub-component of the 2D LiDAR processing pipeline:
1. Scan Validation & Ingestion
2. Coordinate Transformation (Polar to Vehicle Frame)
3. Scan Quality Metrics Evaluation
4. 2D Euclidean Jump-Distance Clustering
5. Multi-Scan Temporal Tracking
6. 2D-to-Model Feature Mapping
7. Frozen ML Risk Inference (HistGradientBoostingClassifier)
8. Total End-to-End Cycle Time

Evaluates performance against the 50.0 ms engineering latency budget.
Generates:
- results/metrics/stage14_benchmark.json
- results/plots/stage14_latency_breakdown.png
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT), str(ROS2_PKG_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.inference.predict import get_inference_engine
from src.perception.feature_mapper_2d import FeatureMapper2D
from src.perception.lidar_2d import LaserScan2D
from src.perception.lidar_calibration_2d import LiDARCalibration2D
from src.perception.obstacle_extractor_2d import (
    MultiScanTracker2D,
    ObstacleExtractor2D,
)
from src.perception.ros2_laserscan_adapter import ROS2LaserScanAdapter
from src.perception.scan_quality_2d import ScanQualityAnalyzer2D
from src.perception.sensor_config_2d import LiDAR2DConfig
from minerakshak_risk.risk_node import PerceptionToRiskProcessor

BENCHMARK_DISCLAIMER = (
    "SOFTWARE BENCHMARK — NOT PHYSICAL SENSOR LATENCY: "
    "Measured on host CPU using synthetic/replay scans. Does not measure physical "
    "2D LiDAR laser pulse time-of-flight, driver polling, or hardware bus latency."
)
ENGINEERING_BUDGET_MS = 50.0


def make_benchmark_scan(
    num_beams: int = 361,
    obstacles: list[tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    """Construct synthetic scan dictionary containing target obstacles.

    obstacles: list of (center_deg, distance_m, angular_width_deg)
    """
    if obstacles is None:
        obstacles = [(0.0, 15.0, 6.0), (-30.0, 25.0, 8.0), (40.0, 35.0, 5.0)]

    ranges = [float("inf")] * num_beams
    intensities = [100.0] * num_beams
    angle_min = -math.pi / 2.0
    angle_max = math.pi / 2.0
    angle_inc = (angle_max - angle_min) / (num_beams - 1)

    for center_deg, dist_m, width_deg in obstacles:
        center_rad = math.radians(center_deg)
        half_w_rad = math.radians(width_deg / 2.0)
        idx_start = int(round((center_rad - half_w_rad - angle_min) / angle_inc))
        idx_end = int(round((center_rad + half_w_rad - angle_min) / angle_inc))
        idx_start = max(0, min(num_beams - 1, idx_start))
        idx_end = max(0, min(num_beams - 1, idx_end))
        for i in range(idx_start, idx_end + 1):
            ranges[i] = dist_m + (i - idx_start) * 0.03

    now_ns = int(time.time() * 1e9)
    return {
        "header": {
            "frame_id": "laser_frame",
            "stamp": {"sec": now_ns // 1_000_000_000, "nanosec": now_ns % 1_000_000_000},
        },
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_inc,
        "time_increment": 0.0,
        "scan_time": 0.05,
        "range_min": 0.1,
        "range_max": 80.0,
        "ranges": ranges,
        "intensities": intensities,
    }


def run_stage14_benchmark(
    warmup_cycles: int = 50,
    benchmark_cycles: int = 200,
) -> dict[str, Any]:
    """Execute rigorous component-wise and total cycle benchmark."""
    config = LiDAR2DConfig(expected_fov_deg=180.0, angular_resolution_deg=0.5)
    calibrator = LiDARCalibration2D(config=config)
    quality_analyzer = ScanQualityAnalyzer2D(config=config)
    adapter = ROS2LaserScanAdapter(config=config, is_synthetic_source=True)
    extractor = ObstacleExtractor2D(cluster_distance_threshold_m=1.2, min_cluster_points=2)
    tracker = MultiScanTracker2D(gating_distance_m=4.5, velocity_smoothing_alpha=0.7)
    mapper = FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5)
    risk_processor = PerceptionToRiskProcessor()

    print(f"[BENCHMARK] Warming up pipeline for {warmup_cycles} iterations...")
    for _ in range(warmup_cycles):
        scan_msg = make_benchmark_scan()
        val_res, scan2d = adapter.ingest_laser_scan(scan_msg)
        if scan2d:
            calibrator.verify_scan_calibration(scan2d)
            quality_analyzer.analyze_scan(scan2d)
            obs = extractor.extract_obstacles(scan2d)
            trk = tracker.track(obs, timestamp_ns=scan2d.timestamp_ns)
            frame = mapper.to_perception_frame(trk, frame_id=scan2d.frame_id, timestamp_ns=scan2d.timestamp_ns)
            risk_processor.process_frame(frame)

    print(f"[BENCHMARK] Executing {benchmark_cycles} timed benchmark iterations...")
    lat_val: list[float] = []
    lat_cal: list[float] = []
    lat_qual: list[float] = []
    lat_clust: list[float] = []
    lat_track: list[float] = []
    lat_map: list[float] = []
    lat_inf: list[float] = []
    lat_total: list[float] = []

    for i in range(benchmark_cycles):
        # Slightly alter obstacle position to simulate dynamic motion
        d_center = 15.0 - (i % 20) * 0.4
        scan_msg = make_benchmark_scan(obstacles=[
            (0.0, d_center, 6.0),
            (-25.0, 22.0, 7.0),
            (35.0, 30.0, 5.0),
        ])

        t0 = time.perf_counter()

        # 1. Ingestion & Format Validation
        t_v0 = time.perf_counter()
        val_res, scan2d = adapter.ingest_laser_scan(scan_msg)
        t_v1 = time.perf_counter()
        lat_val.append((t_v1 - t_v0) * 1000.0)

        # 2. Calibration Verification
        t_c0 = time.perf_counter()
        calibrator.verify_scan_calibration(scan2d)  # type: ignore[arg-type]
        t_c1 = time.perf_counter()
        lat_cal.append((t_c1 - t_c0) * 1000.0)

        # 3. Scan Quality Metrics
        t_q0 = time.perf_counter()
        quality_analyzer.analyze_scan(scan2d)  # type: ignore[arg-type]
        t_q1 = time.perf_counter()
        lat_qual.append((t_q1 - t_q0) * 1000.0)

        # 4. Clustering (2D Euclidean jump-distance)
        t_cl0 = time.perf_counter()
        obstacles = extractor.extract_obstacles(scan2d)  # type: ignore[arg-type]
        t_cl1 = time.perf_counter()
        lat_clust.append((t_cl1 - t_cl0) * 1000.0)

        # 5. Tracking (Multi-scan temporal association)
        t_tr0 = time.perf_counter()
        tracked = tracker.track(obstacles, timestamp_ns=scan2d.timestamp_ns)  # type: ignore[union-attr]
        t_tr1 = time.perf_counter()
        lat_track.append((t_tr1 - t_tr0) * 1000.0)

        # 6. Feature Mapping
        t_m0 = time.perf_counter()
        frame = mapper.to_perception_frame(
            tracked,
            frame_id=scan2d.frame_id,  # type: ignore[union-attr]
            timestamp_ns=scan2d.timestamp_ns,  # type: ignore[union-attr]
            truck_speed_kmph=25.0,
        )
        t_m1 = time.perf_counter()
        lat_map.append((t_m1 - t_m0) * 1000.0)

        # 7. Frozen HGB ML Inference
        t_inf0 = time.perf_counter()
        risk_frame = risk_processor.process_frame(frame)
        t_inf1 = time.perf_counter()
        lat_inf.append((t_inf1 - t_inf0) * 1000.0)

        t_end = time.perf_counter()
        lat_total.append((t_end - t0) * 1000.0)

    def stats_for(arr: list[float]) -> dict[str, float]:
        np_arr = np.array(arr)
        return {
            "mean_ms": float(np.mean(np_arr)),
            "std_ms": float(np.std(np_arr)),
            "median_ms": float(np.median(np_arr)),
            "p95_ms": float(np.percentile(np_arr, 95)),
            "p99_ms": float(np.percentile(np_arr, 99)),
            "min_ms": float(np.min(np_arr)),
            "max_ms": float(np.max(np_arr)),
        }

    results = {
        "stage": "Stage 14",
        "benchmark_title": "2D LiDAR Calibration, Quality, Perception & Inference Latency Benchmark",
        "disclaimer": BENCHMARK_DISCLAIMER,
        "hardware_status": "NOT VALIDATED — HARDWARE NOT AVAILABLE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": benchmark_cycles,
        "engineering_budget_ms": ENGINEERING_BUDGET_MS,
        "budget_passed": bool(np.percentile(lat_total, 99) < ENGINEERING_BUDGET_MS),
        "headroom_pct": float((ENGINEERING_BUDGET_MS - np.mean(lat_total)) / ENGINEERING_BUDGET_MS * 100.0),
        "components": {
            "1_scan_validation": stats_for(lat_val),
            "2_calibration_verification": stats_for(lat_cal),
            "3_scan_quality_metrics": stats_for(lat_qual),
            "4_obstacle_clustering": stats_for(lat_clust),
            "5_multi_scan_tracking": stats_for(lat_track),
            "6_feature_mapping": stats_for(lat_map),
            "7_frozen_hgb_inference": stats_for(lat_inf),
        },
        "total_cycle": stats_for(lat_total),
    }

    # Save JSON report
    out_dir = PROJECT_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage14_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Benchmark metrics saved to: {out_dir / 'stage14_benchmark.json'}")

    # Generate Visualization Plot
    plot_dir = PROJECT_ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generate_benchmark_plot(results, lat_total, plot_dir / "stage14_latency_breakdown.png")

    return results


def generate_benchmark_plot(
    results: dict[str, Any],
    total_latencies: list[float],
    output_path: Path,
) -> None:
    """Generate high-resolution latency breakdown plot."""
    components = results["components"]
    comp_names = [
        "1. Validation",
        "2. Calibration",
        "3. Quality Metrics",
        "4. Clustering",
        "5. Tracking",
        "6. Feature Map",
        "7. Frozen ML Inf",
    ]
    means = [comp["mean_ms"] for comp in components.values()]
    p95s = [comp["p95_ms"] for comp in components.values()]
    p99s = [comp["p99_ms"] for comp in components.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Subplot 1: Latency by Pipeline Component
    x = np.arange(len(comp_names))
    width = 0.25

    rects1 = ax1.bar(x - width, means, width, label="Mean", color="#2563eb", alpha=0.85)
    rects2 = ax1.bar(x, p95s, width, label="P95", color="#f59e0b", alpha=0.85)
    rects3 = ax1.bar(x + width, p99s, width, label="P99", color="#ef4444", alpha=0.85)

    ax1.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Stage 14 Component Latency Breakdown", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_names, rotation=35, ha="right", fontsize=9)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Total Cycle Distribution vs 50ms Budget
    tot_mean = results["total_cycle"]["mean_ms"]
    tot_p95 = results["total_cycle"]["p95_ms"]
    tot_p99 = results["total_cycle"]["p99_ms"]

    n, bins, patches = ax2.hist(total_latencies, bins=25, color="#10b981", alpha=0.75, edgecolor="#047857")
    ax2.axvline(tot_mean, color="#2563eb", linestyle="-", linewidth=2, label=f"Mean: {tot_mean:.2f} ms")
    ax2.axvline(tot_p95, color="#f59e0b", linestyle="--", linewidth=2, label=f"P95: {tot_p95:.2f} ms")
    ax2.axvline(tot_p99, color="#ef4444", linestyle="-.", linewidth=2, label=f"P99: {tot_p99:.2f} ms")
    ax2.axvline(ENGINEERING_BUDGET_MS, color="#7f1d1d", linestyle=":", linewidth=2.5, label="50 ms Budget")

    ax2.set_xlabel("Total Cycle Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.set_title("Total End-to-End Cycle vs Engineering Budget", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Overall Header & Disclaimer Banner
    plt.suptitle(
        "MineRakshak AI — Stage 14 2D LiDAR Pipeline Software Latency Benchmark\n"
        "[SOFTWARE BENCHMARK — NOT PHYSICAL SENSOR LATENCY | PHYSICAL HARDWARE NOT AVAILABLE]",
        fontsize=12,
        fontweight="bold",
        color="#1e293b",
        y=1.03,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Benchmark plot saved to: {output_path}")


if __name__ == "__main__":
    results = run_stage14_benchmark(warmup_cycles=50, benchmark_cycles=200)
    tot = results["total_cycle"]
    print("\n" + "=" * 65)
    print("STAGE 14 SOFTWARE BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Mean Latency:       {tot['mean_ms']:.3f} ms")
    print(f"P95 Latency:        {tot['p95_ms']:.3f} ms")
    print(f"P99 Latency:        {tot['p99_ms']:.3f} ms")
    print(f"Engineering Budget: {ENGINEERING_BUDGET_MS:.1f} ms")
    print(f"Budget Margin:      {results['headroom_pct']:.1f}%")
    print(f"Budget Status:      {'PASSED' if results['budget_passed'] else 'EXCEEDED'}")
    print("=" * 65)
