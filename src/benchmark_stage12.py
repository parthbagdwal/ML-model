"""Stage 12 Benchmark: Synthetic 2D LiDAR Software Benchmark.

===============================================================================
DISCLAIMER:
SYNTHETIC 2D LIDAR SOFTWARE BENCHMARK ONLY
This benchmark measures the software execution latency of the synthetic 2D planar
LiDAR perception pipeline, clustering algorithm, feature compatibility layer,
and frozen HGB risk inference engine.
It does NOT measure physical 2D LiDAR sensor hardware acquisition or hardware bus
transfer timings (e.g. Ethernet / serial / USB).
===============================================================================

Measures:
1. 2D scan conversion & validity filtering latency
2. 2D obstacle extraction & clustering latency
3. Multi-scan tracking & velocity derivation latency
4. Feature mapping & compatibility layer latency
5. Frozen HGB ML risk inference latency
6. End-to-end perception-to-risk cycle latency
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROS2_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PKG_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.inference.predict import get_inference_engine
from src.perception.feature_mapper_2d import (
    STAGE12_COMPATIBILITY_STATEMENT,
    FeatureMapper2D,
)
from src.perception.lidar_2d import LaserScan2D
from src.perception.obstacle_extractor_2d import (
    MultiScanTracker2D,
    ObstacleExtractor2D,
)
from src.perception.synthetic_lidar_2d import (
    SYNTHETIC_2D_LIDAR_DISCLAIMER,
    Synthetic2DLiDARAdapter,
)
from minerakshak_risk.risk_node import PerceptionToRiskProcessor


def run_stage12_benchmark(
    n_iterations: int = 500,
    warmup_iterations: int = 50,
) -> dict[str, Any]:
    """Execute high-precision latency benchmark on synthetic 2D perception pipeline."""
    print("=" * 70)
    print("  STAGE 12: SYNTHETIC 2D LIDAR SOFTWARE BENCHMARK")
    print("=" * 70)
    print(f"Iterations: {n_iterations} (Warmup: {warmup_iterations})")
    print("Sensor Model: Planar 2D LiDAR (180 deg FOV, 0.5 deg beam increment = 361 beams)")
    print("ML Model: Frozen HistGradientBoostingClassifier (12 raw -> 17 transformed features)")
    print("-" * 70)

    # Initialize components
    engine = get_inference_engine()
    adapter = Synthetic2DLiDARAdapter(fov_deg=180.0, resolution_deg=0.5, seed=42)
    extractor = ObstacleExtractor2D(cluster_distance_threshold_m=1.2, min_cluster_points=2)
    tracker = MultiScanTracker2D(gating_distance_m=4.5, velocity_smoothing_alpha=0.7)
    mapper = FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5)
    processor = PerceptionToRiskProcessor(engine=engine)

    # Define standard multi-obstacle test scenario (3 simultaneous obstacles: distant, mid, close)
    test_obstacles = [
        (35.0, math.radians(25.0), 1.2),   # Distant left
        (16.0, math.radians(-20.0), 2.0),  # Mid right
        (6.5, math.radians(2.0), 2.2),     # Close center threat
    ]

    # Warmup
    print("Warming up JIT, caches, and inference engine...")
    for _ in range(warmup_iterations):
        scan = adapter.create_multi_obstacle_scan(obstacles=test_obstacles)
        extracted = extractor.extract_obstacles(scan)
        tracked = tracker.track(extracted, timestamp_ns=scan.timestamp_ns)
        inf_dicts, _ = mapper.map_obstacles_batch(tracked, truck_speed_kmph=25.0)
        _ = engine.predict_batch(inf_dicts)

    tracker.reset()

    # Latency tracking arrays (in milliseconds)
    latencies_point_conv: list[float] = []
    latencies_extraction: list[float] = []
    latencies_tracking: list[float] = []
    latencies_mapping: list[float] = []
    latencies_inference: list[float] = []
    latencies_e2e: list[float] = []

    print(f"Running {n_iterations} timed iterations...")
    base_t_ns = int(time.time() * 1e9)
    dt_ns = int(0.05 * 1e9)  # 20 Hz simulation

    for i in range(n_iterations):
        t_cycle_ns = base_t_ns + (i * dt_ns)

        # 1. Generate multi-obstacle scan
        scan = adapter.create_multi_obstacle_scan(
            obstacles=test_obstacles,
            frame_id=f"bench_scan_{i:04d}",
        )
        scan.timestamp_ns = t_cycle_ns

        t0 = time.perf_counter()

        # Step A: Point conversion & validity check
        ta_start = time.perf_counter()
        pts = scan.to_points()
        ta_end = time.perf_counter()

        # Step B: 2D Obstacle Extraction & Clustering
        tb_start = time.perf_counter()
        extracted_obs = extractor.extract_obstacles(scan)
        tb_end = time.perf_counter()

        # Step C: Multi-Scan Temporal Tracking & Velocity
        tc_start = time.perf_counter()
        tracked_obs = tracker.track(extracted_obs, timestamp_ns=t_cycle_ns)
        tc_end = time.perf_counter()

        # Step D: Feature Compatibility Mapping
        td_start = time.perf_counter()
        inf_dicts, metas = mapper.map_obstacles_batch(tracked_obs, truck_speed_kmph=25.0)
        td_end = time.perf_counter()

        # Step E: Frozen HGB ML Risk Inference
        te_start = time.perf_counter()
        predictions = engine.predict_batch(inf_dicts)
        te_end = time.perf_counter()

        t_end = time.perf_counter()

        latencies_point_conv.append((ta_end - ta_start) * 1000.0)
        latencies_extraction.append((tb_end - tb_start) * 1000.0)
        latencies_tracking.append((tc_end - tc_start) * 1000.0)
        latencies_mapping.append((td_end - td_start) * 1000.0)
        latencies_inference.append((te_end - te_start) * 1000.0)
        latencies_e2e.append((t_end - t0) * 1000.0)

    # Compute statistics
    def calc_stats(data: list[float]) -> dict[str, float]:
        arr = np.array(data)
        return {
            "mean_ms": round(float(np.mean(arr)), 4),
            "std_ms": round(float(np.std(arr)), 4),
            "median_ms": round(float(np.median(arr)), 4),
            "p90_ms": round(float(np.percentile(arr, 90)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "min_ms": round(float(np.min(arr)), 4),
            "max_ms": round(float(np.max(arr)), 4),
        }

    results = {
        "benchmark_label": "Synthetic 2D LiDAR Software Benchmark",
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "disclaimer": SYNTHETIC_2D_LIDAR_DISCLAIMER,
        "compatibility_statement": STAGE12_COMPATIBILITY_STATEMENT,
        "configuration": {
            "scan_fov_deg": 180.0,
            "scan_beam_count": 361,
            "angular_resolution_deg": 0.5,
            "obstacles_per_frame": len(test_obstacles),
            "iterations": n_iterations,
            "warmup_iterations": warmup_iterations,
            "primary_model": "HistGradientBoostingClassifier (FROZEN)",
        },
        "stage_latencies": {
            "point_conversion_and_filtering": calc_stats(latencies_point_conv),
            "obstacle_extraction_and_clustering": calc_stats(latencies_extraction),
            "multi_scan_tracking": calc_stats(latencies_tracking),
            "feature_compatibility_mapping": calc_stats(latencies_mapping),
            "frozen_hgb_model_inference": calc_stats(latencies_inference),
            "end_to_end_pipeline": calc_stats(latencies_e2e),
        },
    }

    # Print summary table
    print("\nBENCHMARK RESULTS (in milliseconds):")
    print("-" * 75)
    print(f"{'Pipeline Stage':<35} | {'Mean':<8} | {'Median':<8} | {'P95':<8} | {'P99':<8}")
    print("-" * 75)
    for stage_name, stats in results["stage_latencies"].items():
        fmt_name = stage_name.replace("_", " ").title()
        print(f"{fmt_name:<35} | {stats['mean_ms']:<8.3f} | {stats['median_ms']:<8.3f} | {stats['p95_ms']:<8.3f} | {stats['p99_ms']:<8.3f}")
    print("-" * 75)

    # Save metrics JSON
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "stage12_2d_benchmark.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Benchmark metrics saved to: {metrics_path}")

    # Generate professional visualization plot
    plots_path = PROJECT_ROOT / "results" / "plots" / "stage12_2d_latency.png"
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    generate_benchmark_plot(results, latencies_e2e, latencies_inference, latencies_extraction, plots_path)
    print(f"[OK] Benchmark latency plot saved to: {plots_path}")

    return results


def generate_benchmark_plot(
    results: dict[str, Any],
    e2e_data: list[float],
    inf_data: list[float],
    ext_data: list[float],
    output_path: Path,
) -> None:
    """Generate high-contrast, publication-quality latency breakdown figure."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Component Mean Latency Breakdown
    stages = [
        "Point Conversion",
        "2D Clustering",
        "Temporal Tracking",
        "Feature Mapping",
        "Frozen HGB Inf",
    ]
    lat_keys = [
        "point_conversion_and_filtering",
        "obstacle_extraction_and_clustering",
        "multi_scan_tracking",
        "feature_compatibility_mapping",
        "frozen_hgb_model_inference",
    ]
    means = [results["stage_latencies"][k]["mean_ms"] for k in lat_keys]
    p95s = [results["stage_latencies"][k]["p95_ms"] for k in lat_keys]

    x = np.arange(len(stages))
    width = 0.38

    ax1 = axes[0]
    rects1 = ax1.bar(x - width / 2, means, width, label="Mean Latency (ms)", color="#1f77b4", alpha=0.9)
    rects2 = ax1.bar(x + width / 2, p95s, width, label="P95 Latency (ms)", color="#ff7f0e", alpha=0.9)

    ax1.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("2D LiDAR Pipeline Component Latency Breakdown", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=20, ha="right", fontsize=9)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Add value annotations
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Plot 2: End-to-End Latency Distribution
    ax2 = axes[1]
    e2e_arr = np.array(e2e_data)
    mean_e2e = float(np.mean(e2e_arr))
    p95_e2e = float(np.percentile(e2e_arr, 95))
    p99_e2e = float(np.percentile(e2e_arr, 99))

    n, bins, patches = ax2.hist(e2e_arr, bins=35, color="#2ca02c", alpha=0.75, edgecolor="black")
    ax2.axvline(mean_e2e, color="blue", linestyle="--", linewidth=1.8, label=f"Mean: {mean_e2e:.2f} ms")
    ax2.axvline(p95_e2e, color="orange", linestyle="--", linewidth=1.8, label=f"P95: {p95_e2e:.2f} ms")
    ax2.axvline(p99_e2e, color="red", linestyle=":", linewidth=2.0, label=f"P99: {p99_e2e:.2f} ms")

    ax2.set_xlabel("End-to-End Pipeline Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.set_title("End-to-End Latency Distribution (500 Scans)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "MineRakshak Stage 12: Synthetic 2D LiDAR Software Benchmark\n"
        "(Deterministic Software Simulation — Does NOT Represent Real Sensor Hardware Latency)",
        fontsize=13, fontweight="bold", y=1.03
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_stage12_benchmark(n_iterations=500, warmup_iterations=50)
