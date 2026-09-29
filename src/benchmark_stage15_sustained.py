"""Stage 15 Sustained Load Performance Benchmark & Fault Recovery Verification.

===============================================================================
MANDATORY BENCHMARK DISCLAIMER:
SOFTWARE/SYNTHETIC BENCHMARK — NOT PHYSICAL HARDWARE LATENCY
This benchmark measures software execution latency on the host CPU over a sustained
10,000-cycle continuous simulation with dynamic obstacles, sensor noise, dropouts,
and corrupted frames. It does NOT measure physical 2D LiDAR time-of-flight, driver
polling, or hardware bus latency. Physical hardware is not connected.
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root, parent directory, and ROS 2 package are on sys.path
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

from minerakshak_risk.laserscan_node import LaserScanToRiskPipeline
from src.perception.sensor_config_2d import (
    LiDAR2DConfig,
    SensorHealthState,
)

BENCHMARK_DISCLAIMER = (
    "SOFTWARE/SYNTHETIC BENCHMARK — NOT PHYSICAL HARDWARE LATENCY: "
    "Measured on host CPU using 10,000 synthetic 2D LaserScan frames. Does not measure "
    "physical 2D LiDAR laser pulse time-of-flight, driver polling, or hardware bus latency."
)
ENGINEERING_BUDGET_MS = 50.0
TOTAL_CYCLES = 10_000
WARMUP_CYCLES = 100


def build_synthetic_scan_frame(
    sim_time_s: float,
    num_beams: int = 361,
    angle_min: float = -math.pi / 2.0,
    angle_max: float = math.pi / 2.0,
    range_min: float = 0.1,
    range_max: float = 80.0,
    cycle_index: int = 0,
) -> tuple[dict[str, Any], str]:
    """Generate dynamic synthetic 2D LaserScan with realistic obstacle kinematics and occasional faults.

    Returns:
        Tuple of (scan_dict, condition_label).
    """
    angle_inc = (angle_max - angle_min) / (num_beams - 1)
    ranges = [float("inf")] * num_beams
    intensities = [100.0] * num_beams
    timestamp_ns = int(sim_time_s * 1e9)
    condition = "nominal"

    # Inject periodic software fault scenarios
    if cycle_index % 500 == 150:
        # Fault: Stale scan timestamp (1.5s in the past)
        timestamp_ns = int((sim_time_s - 1.5) * 1e9)
        condition = "stale_dropout"
    elif cycle_index % 500 == 250:
        # Fault: Negative range corruption
        ranges = [-5.0] * num_beams
        condition = "invalid_negative_ranges"
    elif cycle_index % 500 == 350:
        # Fault: 100% NaN beam corruption
        ranges = [float("nan")] * num_beams
        condition = "invalid_all_nan"
    elif cycle_index % 250 == 80:
        # Fault: Severe beam degradation (90% NaN, e.g. severe dust storm)
        condition = "degraded_dust"
        for i in range(num_beams):
            if i % 10 != 0:
                ranges[i] = float("nan")
        # Keep 1 obstacle in clear sector
        obs_dist = 14.0 + 2.0 * math.sin(sim_time_s * 0.5)
        for i in range(170, 190):
            ranges[i] = obs_dist
    elif cycle_index % 100 == 0:
        # Empty environment (no returns)
        condition = "empty_field"
    else:
        # Nominal Multi-Obstacle Dynamic Scene
        condition = "nominal_multi_obstacle"

        # Obstacle 1: Leading haul truck directly ahead (approaching slowly)
        # Distance oscillates between 28m and 12m with closing rate ~ -2 m/s
        d1 = 20.0 + 8.0 * math.cos(sim_time_s * 0.2)
        idx1_start = int(round((-0.08 - angle_min) / angle_inc))
        idx1_end = int(round((0.08 - angle_min) / angle_inc))
        for i in range(max(0, idx1_start), min(num_beams, idx1_end + 1)):
            ranges[i] = d1 + (i - idx1_start) * 0.02

        # Obstacle 2: Lateral obstacle on left berm (+35 deg)
        # Distance ~ 18m
        ang2 = math.radians(35.0)
        d2 = 18.0
        idx2_start = int(round((ang2 - 0.06 - angle_min) / angle_inc))
        idx2_end = int(round((ang2 + 0.06 - angle_min) / angle_inc))
        for i in range(max(0, idx2_start), min(num_beams, idx2_end + 1)):
            ranges[i] = d2

        # Obstacle 3: Intermittent obstacle on right (-25 deg) appearing and disappearing
        if int(sim_time_s * 0.5) % 2 == 0:
            ang3 = math.radians(-25.0)
            d3 = 8.0 + 4.0 * math.sin(sim_time_s * 0.8)
            idx3_start = int(round((ang3 - 0.07 - angle_min) / angle_inc))
            idx3_end = int(round((ang3 + 0.07 - angle_min) / angle_inc))
            for i in range(max(0, idx3_start), min(num_beams, idx3_end + 1)):
                ranges[i] = d3

    sec = timestamp_ns // 1_000_000_000
    nanosec = timestamp_ns % 1_000_000_000

    scan_msg = {
        "header": {
            "frame_id": "laser_frame",
            "stamp": {"sec": sec, "nanosec": nanosec},
        },
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_inc,
        "time_increment": 0.0,
        "scan_time": 0.05,
        "range_min": range_min,
        "range_max": range_max,
        "ranges": ranges,
        "intensities": intensities,
    }

    return scan_msg, condition


def run_sustained_benchmark(num_cycles: int = TOTAL_CYCLES) -> dict[str, Any]:
    """Execute the sustained 10,000-cycle end-to-end benchmark."""
    print("=" * 80)
    print("MINERAKSHAK AI — STAGE 15 SUSTAINED PERFORMANCE & FAULT RECOVERY BENCHMARK")
    print(f"Cycles: {num_cycles:,} | Engineering Budget: {ENGINEERING_BUDGET_MS:.1f} ms")
    print("DISCLAIMER: SOFTWARE/SYNTHETIC BENCHMARK — NOT PHYSICAL HARDWARE LATENCY")
    print("=" * 80)

    # Initialize production pipeline
    cfg = LiDAR2DConfig(stale_timeout_s=0.5)
    pipeline = LaserScanToRiskPipeline(config=cfg, default_truck_speed_kmph=28.0)

    # 1. Warm-up Phase
    print(f"\n[1/3] Running {WARMUP_CYCLES} warm-up cycles...")
    base_time = 1000.0
    for w in range(WARMUP_CYCLES):
        scan, _ = build_synthetic_scan_frame(sim_time_s=base_time + w * 0.05, cycle_index=w)
        pipeline.process_scan(scan, current_time_s=base_time + w * 0.05)
    pipeline.reset_tracker()
    print("      Warm-up complete. Cache primed, tracker reset.")

    # 2. Sustained Benchmark Execution
    print(f"\n[2/3] Executing sustained {num_cycles:,} cycles...")
    tracemalloc.start()
    mem_start_b, _ = tracemalloc.get_traced_memory()

    latencies_ms: list[float] = []
    health_counts: dict[str, int] = {s.value: 0 for s in SensorHealthState}
    threat_counts: dict[str, int] = {"SAFE": 0, "CAUTION": 0, "WARNING": 0, "CRITICAL": 0}
    condition_counts: dict[str, int] = {}
    health_transitions = 0
    last_health_state: str = SensorHealthState.NO_DATA.value
    faults_isolated = 0

    t_bench_start = time.perf_counter()

    for cycle in range(num_cycles):
        sim_time = base_time + (cycle * 0.05)
        scan_msg, condition = build_synthetic_scan_frame(
            sim_time_s=sim_time,
            cycle_index=cycle,
        )
        condition_counts[condition] = condition_counts.get(condition, 0) + 1

        t0 = time.perf_counter()
        val_res, risk_frame = pipeline.process_scan(
            scan_msg,
            truck_speed_kmph=28.0,
            current_time_s=sim_time,
        )
        t_elap_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_elap_ms)

        # Track health state
        curr_health = val_res.health_state.value
        health_counts[curr_health] += 1
        if curr_health != last_health_state:
            health_transitions += 1
            last_health_state = curr_health

        # Track threat level
        curr_threat = risk_frame.highest_threat_level
        threat_counts[curr_threat] = threat_counts.get(curr_threat, 0) + 1

        if not val_res.is_valid or risk_frame.errors:
            faults_isolated += 1

        if (cycle + 1) % 2000 == 0 or (cycle + 1) == num_cycles:
            avg_so_far = np.mean(latencies_ms)
            p99_so_far = np.percentile(latencies_ms, 99)
            print(
                f"      Cycle {cycle + 1:,} / {num_cycles:,} | "
                f"Mean: {avg_so_far:.3f} ms | P99: {p99_so_far:.3f} ms | "
                f"Faults Isolated: {faults_isolated}"
            )

    total_bench_duration_s = time.perf_counter() - t_bench_start
    mem_current_b, mem_peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_start_mb = mem_start_b / (1024 * 1024)
    mem_current_mb = mem_current_b / (1024 * 1024)
    mem_peak_mb = mem_peak_b / (1024 * 1024)
    mem_growth_mb = max(0.0, mem_current_mb - mem_start_mb)

    # Compute Statistical Metrics
    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    min_lat = float(np.min(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    max_lat = float(np.max(lat_arr))
    throughput_fps = float(num_cycles / total_bench_duration_s)

    budget_margin_ms = ENGINEERING_BUDGET_MS - mean_lat
    budget_passed = p99_lat < ENGINEERING_BUDGET_MS

    print("\n" + "=" * 80)
    print("SUSTAINED BENCHMARK RESULTS SUMMARY (10,000 CYCLES)")
    print("=" * 80)
    print(f"Total Cycles Processed:    {num_cycles:,}")
    print(f"Total Benchmark Duration:  {total_bench_duration_s:.3f} s")
    print(f"Throughput:                {throughput_fps:.1f} FPS")
    print("-" * 50)
    print(f"Mean Software Latency:     {mean_lat:.3f} ms (std: {std_lat:.3f} ms)")
    print(f"Minimum Latency:           {min_lat:.3f} ms")
    print(f"P50 Latency (Median):      {p50_lat:.3f} ms")
    print(f"P90 Latency:               {p90_lat:.3f} ms")
    print(f"P95 Latency:               {p95_lat:.3f} ms")
    print(f"P99 Latency:               {p99_lat:.3f} ms")
    print(f"Maximum Latency:           {max_lat:.3f} ms")
    print(f"Engineering Budget:        {ENGINEERING_BUDGET_MS:.1f} ms")
    print(f"Safety Margin (Budget-Mean): {budget_margin_ms:.3f} ms ({budget_margin_ms/ENGINEERING_BUDGET_MS*100:.1f}%)")
    print(f"Budget Verification:       {'PASS [OK]' if budget_passed else 'FAIL'}")
    print("-" * 50)
    print(f"Faults Isolated Safely:    {faults_isolated:,}")
    print(f"Health State Transitions:  {health_transitions:,}")
    print(f"Health Distribution:       {health_counts}")
    print(f"Threat Tier Distribution:  {threat_counts}")
    print(f"Memory (Start / Peak / Growth): {mem_start_mb:.2f} MB / {mem_peak_mb:.2f} MB / +{mem_growth_mb:.2f} MB")
    print("=" * 80)

    # Assemble Structured Report
    report = {
        "stage": "Stage 15",
        "benchmark_title": "MineRakshak AI — Sustained Software Load Benchmark & Fault Recovery Verification",
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "disclaimer": BENCHMARK_DISCLAIMER,
        "is_physical_hardware_validated": False,
        "physical_hardware_status": "NOT_AVAILABLE — Simulation & synthetic input only",
        "configuration": {
            "total_cycles": num_cycles,
            "warmup_cycles": WARMUP_CYCLES,
            "simulated_sensor_fov_deg": 180.0,
            "simulated_beam_count": 361,
            "engineering_latency_budget_ms": ENGINEERING_BUDGET_MS,
            "primary_model": "HistGradientBoostingClassifier (FROZEN)",
            "temporal_tracker": "MultiScanTracker2D(gating=4.5m, alpha=0.7)",
        },
        "latency_metrics_ms": {
            "mean": round(mean_lat, 4),
            "std": round(std_lat, 4),
            "min": round(min_lat, 4),
            "p50": round(p50_lat, 4),
            "p90": round(p90_lat, 4),
            "p95": round(p95_lat, 4),
            "p99": round(p99_lat, 4),
            "max": round(max_lat, 4),
            "budget_ms": ENGINEERING_BUDGET_MS,
            "budget_margin_ms": round(budget_margin_ms, 4),
            "budget_compliance": "PASS" if budget_passed else "FAIL",
        },
        "throughput": {
            "total_duration_s": round(total_bench_duration_s, 3),
            "frames_per_second": round(throughput_fps, 2),
        },
        "fault_and_health_metrics": {
            "total_faults_isolated": faults_isolated,
            "total_health_transitions": health_transitions,
            "sensor_health_distribution": health_counts,
            "threat_level_distribution": threat_counts,
            "simulated_condition_distribution": condition_counts,
        },
        "memory_metrics_mb": {
            "start_mb": round(mem_start_mb, 2),
            "peak_mb": round(mem_peak_mb, 2),
            "net_growth_mb": round(mem_growth_mb, 2),
        },
    }

    # Save Metrics JSON
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "stage15_sustained_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[3/3] Benchmark metrics saved: {json_path}")

    # Generate Visualization Plot
    plots_dir = PROJECT_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "stage15_sustained_performance.png"
    _generate_benchmark_plots(latencies_ms, health_counts, threat_counts, plot_path)
    print(f"      Performance plot saved: {plot_path}")

    return report


def _generate_benchmark_plots(
    latencies: list[float],
    health_counts: dict[str, int],
    threat_counts: dict[str, int],
    save_path: Path,
) -> None:
    """Generate comprehensive 3-panel performance diagnostic plot."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [1.2, 1.2, 0.9]})

    # Panel 1: Latency Distribution Histogram
    ax1 = axes[0]
    lat_arr = np.array(latencies)
    mean_val = np.mean(lat_arr)
    p50_val = np.percentile(lat_arr, 50)
    p95_val = np.percentile(lat_arr, 95)
    p99_val = np.percentile(lat_arr, 99)

    ax1.hist(lat_arr, bins=80, color="#1976D2", edgecolor="#0D47A1", alpha=0.75, label="Latency Distribution")
    ax1.axvline(mean_val, color="#FF9800", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f} ms")
    ax1.axvline(p95_val, color="#E91E63", linestyle="--", linewidth=2, label=f"P95: {p95_val:.2f} ms")
    ax1.axvline(p99_val, color="#D32F2F", linestyle="-", linewidth=2.5, label=f"P99: {p99_val:.2f} ms")
    ax1.axvline(ENGINEERING_BUDGET_MS, color="#B71C1C", linestyle=":", linewidth=2.5, label="50 ms Budget Limit")

    ax1.set_title("Stage 15: Software Latency Distribution (10,000 Cycles)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Cycle Latency (ms)", fontsize=11)
    ax1.set_ylabel("Frame Count", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Latency Timeline & Rolling Trend
    ax2 = axes[1]
    window = 100
    rolling_mean = np.convolve(lat_arr, np.ones(window) / window, mode="valid")
    x_rolling = np.arange(window // 2, len(rolling_mean) + window // 2)

    ax2.plot(lat_arr, color="#90CAF9", alpha=0.35, linewidth=0.5, label="Per-Cycle Latency")
    ax2.plot(x_rolling, rolling_mean, color="#0D47A1", linewidth=2.0, label=f"{window}-Cycle Rolling Average")
    ax2.axhline(ENGINEERING_BUDGET_MS, color="#B71C1C", linestyle=":", linewidth=2.0, label="50 ms Budget")

    ax2.set_title("Stage 15: 10,000-Cycle Sustained Latency Timeline", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Simulation Cycle Index", fontsize=11)
    ax2.set_ylabel("Latency (ms)", fontsize=11)
    ax2.set_ylim(0, min(max(lat_arr) * 1.15, 60.0))
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Operational Health & Risk Breakdown
    ax3 = axes[2]
    categories = list(health_counts.keys()) + list(threat_counts.keys())
    values = list(health_counts.values()) + list(threat_counts.values())
    colors = ["#4CAF50", "#9E9E9E", "#FFC107", "#F44336", "#FF9800", "#4CAF50", "#FFEB3B", "#FF9800", "#D32F2F"]

    bars = ax3.bar(categories, values, color=colors[:len(categories)], edgecolor="#424242", alpha=0.85)
    ax3.set_title("Operational Sensor Health & Evaluated Threat Level Distribution", fontsize=13, fontweight="bold", pad=10)
    ax3.set_ylabel("Cycle Count", fontsize=11)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax3.annotate(
                f"{h:,}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    run_sustained_benchmark(TOTAL_CYCLES)
