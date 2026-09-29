"""Stage 8: Executable Smoke Test & Latency Verification for MineRakshak Inference Pipeline.

Executes:
1. Model and preprocessor artifact loading.
2. Single-sample inference for SAFE and CRITICAL obstacles.
3. Unknown object type handling.
4. Input validation and error catching.
5. Latency benchmarking over warm-up cycles (mean, P50, P95, P99, max).
6. Exports results to results/metrics/inference_smoke_test.json.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.inference.predict import (
    InferenceValidationError,
    MineRakshakInferenceEngine,
    SAFETY_DISCLAIMER,
    predict,
)
from src.preprocessing.schema import RISK_CLASSES

OUTPUT_JSON_PATH = PROJECT_ROOT / "results" / "metrics" / "inference_smoke_test.json"


def run_smoke_test() -> dict:
    """Run end-to-end smoke test and latency benchmark."""
    print("=" * 70)
    print("mineRakshak-ai: Stage 8 Production Inference Smoke Test")
    print("=" * 70)

    # 1. Initialize Engine
    t_start = time.perf_counter()
    engine = MineRakshakInferenceEngine(load_secondary=True)
    load_time_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"[OK] Artifacts loaded in {load_time_ms:.2f} ms")

    # 2. Test Samples
    safe_sample = {
        "distance_m": 72.0,
        "object_x_m": 5.0,
        "object_y_m": 71.8,
        "object_z_m": 0.0,
        "object_width_m": 2.2,
        "object_height_m": 1.6,
        "object_length_m": 4.5,
        "point_count": 90,
        "relative_velocity_mps": 0.0,
        "truck_speed_kmph": 18.0,
        "time_to_collision_s": 99.9,
        "object_type": "car",
    }

    critical_sample = {
        "distance_m": 6.0,
        "object_x_m": 1.0,
        "object_y_m": 5.9,
        "object_z_m": 0.4,
        "object_width_m": 3.4,
        "object_height_m": 3.6,
        "object_length_m": 7.2,
        "point_count": 780,
        "relative_velocity_mps": -9.2,
        "truck_speed_kmph": 30.0,
        "time_to_collision_s": 0.65,
        "object_type": "truck",
    }

    # 3. Evaluate Predictions
    pred_safe = engine.predict_single(safe_sample)
    pred_critical = engine.predict_single(critical_sample)

    print(f"SAFE sample prediction:     {pred_safe['predicted_risk_level']} (Conf: {pred_safe['confidence']:.4f})")
    print(f"CRITICAL sample prediction: {pred_critical['predicted_risk_level']} (Conf: {pred_critical['confidence']:.4f})")

    assert pred_safe["predicted_risk_level"] == "SAFE", "SAFE sample did not predict SAFE"
    assert pred_critical["predicted_risk_level"] == "CRITICAL", "CRITICAL sample did not predict CRITICAL"

    # 4. Unknown Object Type Handling
    unknown_sample = {**safe_sample, "object_type": "novel_autonomous_drone"}
    pred_unknown = engine.predict_single(unknown_sample)
    print(f"Unknown type prediction:   {pred_unknown['predicted_risk_level']} (Conf: {pred_unknown['confidence']:.4f})")
    assert pred_unknown["predicted_risk_level"] in RISK_CLASSES

    # 5. Validation Error Handling
    validation_passed = False
    try:
        engine.predict_single({**critical_sample, "distance_m": -5.0})
    except InferenceValidationError:
        validation_passed = True
    assert validation_passed, "Validation error was not raised for negative distance"
    print("[OK] Input validation correctly rejected negative distance")

    # 6. Latency Profiling (Warmup + Benchmark)
    warmup_iters = 50
    bench_iters = 500

    for _ in range(warmup_iters):
        engine.predict_single(safe_sample)

    latencies_ms: list[float] = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        engine.predict_single(critical_sample)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    max_lat = float(np.max(latencies_ms))
    min_lat = float(np.min(latencies_ms))

    print("\n--- Latency Benchmark (500 iterations) ---")
    print(f"Mean Latency:   {mean_lat:.3f} ms")
    print(f"Median (P50):   {median_lat:.3f} ms")
    print(f"P95 Latency:    {p95_lat:.3f} ms")
    print(f"P99 Latency:    {p99_lat:.3f} ms")
    print(f"Max Latency:    {max_lat:.3f} ms")
    print(f"Meets < 50ms:   {mean_lat < 50.0}")

    smoke_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke_test_status": "PASSED",
        "artifacts_loaded": {
            "primary_model": "models/hgb_model.joblib",
            "secondary_model": "models/xgboost_model.json",
            "preprocessor": "models/preprocessor.joblib",
            "feature_schema": "models/feature_schema.json",
            "label_mapping": "models/label_mapping.json",
            "load_time_ms": round(load_time_ms, 2),
        },
        "example_predictions": {
            "safe_sample": {
                "input": safe_sample,
                "predicted_risk_level": pred_safe["predicted_risk_level"],
                "risk_code": pred_safe["risk_code"],
                "confidence": round(pred_safe["confidence"], 4),
                "probabilities": {k: round(v, 4) for k, v in pred_safe["probabilities"].items()},
                "recommended_action": pred_safe["recommended_action"],
            },
            "critical_sample": {
                "input": critical_sample,
                "predicted_risk_level": pred_critical["predicted_risk_level"],
                "risk_code": pred_critical["risk_code"],
                "confidence": round(pred_critical["confidence"], 4),
                "probabilities": {k: round(v, 4) for k, v in pred_critical["probabilities"].items()},
                "recommended_action": pred_critical["recommended_action"],
            },
        },
        "latency_profile_ms": {
            "iterations": bench_iters,
            "warmup_iterations": warmup_iters,
            "mean_ms": round(mean_lat, 4),
            "median_p50_ms": round(median_lat, 4),
            "p95_ms": round(p95_lat, 4),
            "p99_ms": round(p99_lat, 4),
            "min_ms": round(min_lat, 4),
            "max_ms": round(max_lat, 4),
            "meets_50ms_target": mean_lat < 50.0,
            "margin_ms": round(50.0 - mean_lat, 4),
        },
        "disclaimer": SAFETY_DISCLAIMER,
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(smoke_data, f, indent=2)

    print(f"\n[OK] Smoke test metrics written to: {OUTPUT_JSON_PATH}")
    return smoke_data


if __name__ == "__main__":
    run_smoke_test()
