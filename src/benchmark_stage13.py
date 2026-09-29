"""Stage 13 Latency Benchmark: Hardware-Ready 2D LiDAR Pipeline.

===============================================================================
MANDATORY BENCHMARK DISCLAIMER:
SYNTHETIC SOFTWARE BENCHMARK — NOT PHYSICAL HARDWARE LATENCY
This benchmark measures the software execution latency of the hardware-ready
2D planar LiDAR ROS 2 interface, scan validation, coordinate conversion,
obstacle extraction, tracking, feature mapping, and frozen ML risk inference.
It does NOT measure physical 2D LiDAR sensor hardware acquisition, spinning mirror
time, or physical bus transmission latencies (Ethernet / USB / Serial) because
no physical sensor hardware is currently connected.
===============================================================================

Measures separately:
1. LaserScan validation latency
2. LaserScan conversion (polar-to-Cartesian + mounting offsets) latency
3. Obstacle extraction & clustering latency
4. Multi-scan temporal tracking latency
5. Feature mapping to frozen model contract latency
6. Frozen HGB ML risk inference latency
7. Total end-to-end software cycle latency
Compared against the 50 ms real-time mining safety budget (20 Hz requirement).
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

for p in [str(PROJECT_ROOT), str(ROS2_PKG_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
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
from src.perception.ros2_laserscan_adapter import (
    HARDWARE_READY_DISCLAIMER,
    ROS2LaserScanAdapter,
)
from src.perception.sensor_config_2d import LiDAR2DConfig
from src.perception.synthetic_lidar_2d import Synthetic2DLiDARAdapter
from minerakshak_risk.laserscan_node import LaserScanToRiskPipeline

DISCLAIMER: str = "SYNTHETIC SOFTWARE BENCHMARK — NOT PHYSICAL HARDWARE LATENCY"
REALTIME_BUDGET_MS: float = 50.0  # 20 Hz cycle budget


def run_stage13_benchmark(
    n_iterations: int = 500,
    warmup_iterations: int = 50,
) -> dict[str, Any]:
    """Execute high-precision latency benchmark on the hardware-ready 2D LiDAR pipeline."""
    print("=" * 76)
    print("  STAGE 13: REAL 2D LIDAR ROS 2 INTERFACE & PIPELINE BENCHMARK")
    print(f"  {DISCLAIMER}")
    print("=" * 76)
    print(f"Iterations: {n_iterations} (Warmup: {warmup_iterations})")
    print("Input: Simulated ROS 2 LaserScan message (361 beams @ 0.5 deg resolution)")
    print("ML Model: Frozen HistGradientBoostingClassifier (models/hgb_model.joblib)")
    print(f"Engineering Latency Budget: {REALTIME_BUDGET_MS:.1f} ms (20 Hz)")
    print("-" * 76)

    # Initialize components
    engine = get_inference_engine()
    cfg = LiDAR2DConfig(
        mounting_yaw_offset_rad=0.0,
        mounting_x_offset_m=0.0,
        mounting_y_offset_m=0.0,
        stale_timeout_s=5.0,  # relaxed for benchmark run
    )
    synthetic_adapter = Synthetic2DLiDARAdapter(fov_deg=180.0, resolution_deg=0.5, seed=42)
    adapter = ROS2LaserScanAdapter(config=cfg, is_synthetic_source=True)
    extractor = ObstacleExtractor2D(cluster_distance_threshold_m=1.2, min_cluster_points=2)
    tracker = MultiScanTracker2D(gating_distance_m=4.5, velocity_smoothing_alpha=0.7)
    mapper = FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5)

    # Standard benchmark scenario: 3 distinct obstacles (distant left, mid right, imminent center)
    test_obstacles = [
        (35.0, math.radians(25.0), 1.2),   # Distant left
        (16.0, math.radians(-20.0), 2.0),  # Mid right
        (6.5, math.radians(2.0), 2.2),     # Close center threat
    ]
    base_scan = synthetic_adapter.create_multi_obstacle_scan(obstacles=test_obstacles)
    ros2_dict = base_scan.to_dict()
    ros2_dict["header"] = {
        "frame_id": "laser_frame",
        "stamp": {"sec": 1000, "nanosec": 0},
    }

    # Warmup
    print("Executing warmup cycles...")
    for i in range(warmup_iterations):
        ros2_dict["header"]["stamp"]["sec"] = 1000 + i
        val_res, scan2d = adapter.ingest_laser_scan(ros2_dict, current_time_s=1000.0 + i + 0.01)
        if scan2d:
            _ = scan2d.to_points()
            obs = extractor.extract_obstacles(scan2d)
            trk = tracker.track(obs, timestamp_ns=scan2d.timestamp_ns)
            pf = mapper.to_perception_frame(trk, truck_speed_kmph=25.0, frame_id=scan2d.frame_id, timestamp_ns=scan2d.timestamp_ns)
            inf_dicts, _ = mapper.map_obstacles_batch(trk, truck_speed_kmph=25.0)
            _ = engine.predict_batch(inf_dicts)

    tracker.reset()

    # Latency tracking arrays (milliseconds)
    latencies_val: list[float] = []
    latencies_conv: list[float] = []
    latencies_ext: list[float] = []
    latencies_trk: list[float] = []
    latencies_map: list[float] = []
    latencies_inf: list[float] = []
    latencies_total: list[float] = []

    print(f"Benchmarking {n_iterations} cycles across all 6 stages...")
    for i in range(n_iterations):
        sim_sec = 2000 + (i * 0.05)
        ros2_dict["header"]["stamp"]["sec"] = int(sim_sec)
        ros2_dict["header"]["stamp"]["nanosec"] = int((sim_sec - int(sim_sec)) * 1e9)
        ref_time = sim_sec + 0.01

        t0 = time.perf_counter()

        # Stage 1: LaserScan Validation
        t_v0 = time.perf_counter()
        val_res, scan2d = adapter.ingest_laser_scan(ros2_dict, current_time_s=ref_time)
        t_v1 = time.perf_counter()
        latencies_val.append((t_v1 - t_v0) * 1000.0)

        # Stage 2: LaserScan Conversion (polar to vehicle Cartesian + offsets)
        t_c0 = time.perf_counter()
        points = scan2d.to_points()
        t_c1 = time.perf_counter()
        latencies_conv.append((t_c1 - t_c0) * 1000.0)

        # Stage 3: Obstacle Extraction & Clustering
        t_e0 = time.perf_counter()
        obstacles = extractor.extract_obstacles(scan2d)
        t_e1 = time.perf_counter()
        latencies_ext.append((t_e1 - t_e0) * 1000.0)

        # Stage 4: Multi-Scan Tracking & Kinematic Velocity
        t_t0 = time.perf_counter()
        tracked = tracker.track(obstacles, timestamp_ns=scan2d.timestamp_ns)
        t_t1 = time.perf_counter()
        latencies_trk.append((t_t1 - t_t0) * 1000.0)

        # Stage 5: Feature Mapping to Model Schema Contract
        t_m0 = time.perf_counter()
        inf_dicts, _ = mapper.map_obstacles_batch(tracked, truck_speed_kmph=25.0)
        t_m1 = time.perf_counter()
        latencies_map.append((t_m1 - t_m0) * 1000.0)

        # Stage 6: Frozen HGB ML Risk Inference
        t_i0 = time.perf_counter()
        _ = engine.predict_batch(inf_dicts)
        t_i1 = time.perf_counter()
        latencies_inf.append((t_i1 - t_i0) * 1000.0)

        t_end = time.perf_counter()
        latencies_total.append((t_end - t0) * 1000.0)

    # Compute Statistics
    def stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    val_s = stats(latencies_val)
    conv_s = stats(latencies_conv)
    ext_s = stats(latencies_ext)
    trk_s = stats(latencies_trk)
    map_s = stats(latencies_map)
    inf_s = stats(latencies_inf)
    tot_s = stats(latencies_total)

    budget_consumption_pct = (tot_s["mean"] / REALTIME_BUDGET_MS) * 100.0

    # Display Results Table
    print("\n" + "=" * 76)
    print(f"{'Pipeline Stage':<28} | {'Mean (ms)':<10} | {'Median':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10}")
    print("-" * 76)
    print(f"{'1. LaserScan Validation':<28} | {val_s['mean']:<10.3f} | {val_s['median']:<10.3f} | {val_s['p95']:<10.3f} | {val_s['p99']:<10.3f}")
    print(f"{'2. LaserScan Conversion':<28} | {conv_s['mean']:<10.3f} | {conv_s['median']:<10.3f} | {conv_s['p95']:<10.3f} | {conv_s['p99']:<10.3f}")
    print(f"{'3. Obstacle Extraction':<28} | {ext_s['mean']:<10.3f} | {ext_s['median']:<10.3f} | {ext_s['p95']:<10.3f} | {ext_s['p99']:<10.3f}")
    print(f"{'4. Multi-Scan Tracking':<28} | {trk_s['mean']:<10.3f} | {trk_s['median']:<10.3f} | {trk_s['p95']:<10.3f} | {trk_s['p99']:<10.3f}")
    print(f"{'5. Feature Mapping':<28} | {map_s['mean']:<10.3f} | {map_s['median']:<10.3f} | {map_s['p95']:<10.3f} | {map_s['p99']:<10.3f}")
    print(f"{'6. Frozen HGB Inference':<28} | {inf_s['mean']:<10.3f} | {inf_s['median']:<10.3f} | {inf_s['p95']:<10.3f} | {inf_s['p99']:<10.3f}")
    print("-" * 76)
    print(f"{'TOTAL SOFTWARE CYCLE':<28} | {tot_s['mean']:<10.3f} | {tot_s['median']:<10.3f} | {tot_s['p95']:<10.3f} | {tot_s['p99']:<10.3f}")
    print("=" * 76)
    print(f"Engineering Budget: {REALTIME_BUDGET_MS:.1f} ms | Total Cycle: {tot_s['mean']:.2f} ms ({budget_consumption_pct:.1f}% budget consumed)")
    verdict = "PASSED" if tot_s["p99"] < REALTIME_BUDGET_MS else "FAILED"
    print(f"Real-Time Verdict: {verdict} (P99 {tot_s['p99']:.2f} ms < {REALTIME_BUDGET_MS:.1f} ms)")
    print("=" * 76)

    # Save Metrics JSON
    metrics_data = {
        "stage": 13,
        "benchmark_name": "Stage 13 Real 2D LiDAR Software Latency Benchmark",
        "benchmark_disclaimer": DISCLAIMER,
        "hardware_validated": False,
        "n_iterations": n_iterations,
        "beams_per_scan": 361,
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "budget_consumption_pct": round(budget_consumption_pct, 2),
        "verdict": verdict,
        "stages": {
            "validation_ms": val_s,
            "conversion_ms": conv_s,
            "extraction_ms": ext_s,
            "tracking_ms": trk_s,
            "mapping_ms": map_s,
            "inference_ms": inf_s,
            "total_cycle_ms": tot_s,
        },
    }

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "stage13_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\n[OK] Benchmark metrics saved to: {json_path}")

    # Generate and Save Latency Breakdown Plot
    plots_dir = PROJECT_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "stage13_latency_breakdown.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")

    # Bar chart of mean stage latencies
    stage_names = [
        "1. Validation",
        "2. Conversion",
        "3. Extraction",
        "4. Tracking",
        "5. Mapping",
        "6. ML Inf.",
    ]
    means = [val_s["mean"], conv_s["mean"], ext_s["mean"], trk_s["mean"], map_s["mean"], inf_s["mean"]]
    colors = ["#38bdf8", "#818cf8", "#a78bfa", "#f472b6", "#fb923c", "#34d399"]

    bars = ax1.bar(stage_names, means, color=colors, edgecolor="#e2e8f0", linewidth=0.8)
    ax1.set_title("Mean Latency by Pipeline Stage (ms)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Latency (ms)", color="#cbd5e1", fontsize=10)
    ax1.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     color="#f8fafc", fontsize=8)

    # Boxplot of total cycle latency vs 50ms budget
    bp = ax2.boxplot(
        [latencies_total],
        patch_artist=True,
        boxprops=dict(facecolor="#0284c7", color="#f8fafc"),
        whiskerprops=dict(color="#f8fafc"),
        capprops=dict(color="#f8fafc"),
        medianprops=dict(color="#f59e0b", linewidth=2.0),
        flierprops=dict(marker="o", markerfacecolor="#f43f5e", markersize=4),
    )
    ax2.axhline(REALTIME_BUDGET_MS, color="#ef4444", linestyle="--", linewidth=1.8, label=f"50ms Budget ({REALTIME_BUDGET_MS:.0f} ms)")
    ax2.set_title(f"Total Software Cycle vs. 50ms Budget ({tot_s['mean']:.2f} ms Mean)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax2.set_ylabel("Total Latency (ms)", color="#cbd5e1", fontsize=10)
    ax2.set_xticklabels(["Hardware-Ready Path"], color="#cbd5e1", fontsize=9.5)
    ax2.tick_params(colors="#cbd5e1")
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", loc="upper right")

    fig.suptitle(
        f"Stage 13 2D LiDAR Software Latency Benchmark (N={n_iterations})\n"
        f"[{DISCLAIMER}]",
        color="#f8fafc",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(plot_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Benchmark plot saved to: {plot_path}")

    return metrics_data


if __name__ == "__main__":
    run_stage13_benchmark(n_iterations=500, warmup_iterations=50)
