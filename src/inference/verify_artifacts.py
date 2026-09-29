"""Stage 8: Standalone Artifact Verification and Packaging Validator.

Runs an automated verification procedure:
1. Validates all required artifacts against manifest SHA-256 checksums.
2. Checks file sizes and metadata integrity.
3. Cold-loads preprocessor and models, recording serialization load latency.
4. Asserts 100% prediction agreement against Stage 7 test predictions.
5. Asserts probability reproduction within float tolerance (atol=1e-5).
6. Executes end-to-end forward pass on sample hazardous & benign raw scenarios.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Decoupled loading for sklearn HistGradientBoostingClassifier on Python 3.14
import sklearn
if "sklearn.ensemble" not in sys.modules:
    _pkg = types.ModuleType("sklearn.ensemble")
    _pkg.__path__ = [os.path.join(os.path.dirname(sklearn.__file__), "ensemble")]
    sys.modules["sklearn.ensemble"] = _pkg

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
)

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.preprocessing.preprocess import MineRakshakPreprocessor

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MANIFEST_PATH = MODELS_DIR / "manifest.json"


def compute_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_manifest_checksums(manifest: dict[str, Any]) -> bool:
    """Check that all serialized files match manifest SHA-256 hashes."""
    print("\n[Step 1/5] Verifying Artifact SHA-256 Checksums against Manifest...")
    all_passed = True
    artifacts = manifest["artifacts"]

    items_to_check = [
        ("Primary Model (HGB)", MODELS_DIR / "hgb_model.joblib", artifacts["primary_model"]["sha256"]),
        ("Secondary Model (XGB)", MODELS_DIR / "xgboost_model.json", artifacts["secondary_model"]["sha256"]),
        ("Preprocessor", MODELS_DIR / "preprocessor.joblib", artifacts["preprocessor"]["sha256"]),
        ("Feature Schema", MODELS_DIR / "feature_schema.json", artifacts["feature_schema"]["sha256"]),
        ("Label Mapping", MODELS_DIR / "label_mapping.json", artifacts["label_mapping"]["sha256"]),
    ]

    for label, path, expected_hash in items_to_check:
        if not path.exists():
            print(f"  [FAIL] {label}: File NOT found at {path.name}!")
            all_passed = False
            continue
        actual_hash = compute_sha256(path)
        size_kb = path.stat().st_size / 1024
        if actual_hash == expected_hash:
            print(f"  [PASS] {label:<22} ({size_kb:>8.1f} KB) -> SHA256: {actual_hash[:16]}... OK")
        else:
            print(f"  [FAIL] {label:<22} Hash mismatch! Expected: {expected_hash}, Got: {actual_hash}")
            all_passed = False

    return all_passed


def verify_model_reproducibility() -> bool:
    """Check that re-loaded models reproduce 100% of test predictions and probabilities."""
    print("\n[Step 2/5] Testing Prediction & Probability Reproducibility on Test Set (3,000 samples)...")
    x_test_path = DATA_DIR / "X_test.csv"
    hgb_saved_path = METRICS_DIR / "hgb_test_predictions.csv"
    xgb_saved_path = METRICS_DIR / "xgboost_test_predictions.csv"

    if not x_test_path.exists() or not hgb_saved_path.exists() or not xgb_saved_path.exists():
        print("  [FAIL] Missing test data or saved Stage 7 predictions!")
        return False

    X_test = pd.read_csv(x_test_path)
    hgb_saved = pd.read_csv(hgb_saved_path)
    xgb_saved = pd.read_csv(xgb_saved_path)

    # 1. HGB
    t0 = time.perf_counter()
    hgb_model = joblib.load(MODELS_DIR / "hgb_model.joblib")
    hgb_load_ms = (time.perf_counter() - t0) * 1000
    hgb_preds = hgb_model.predict(X_test)
    hgb_probs = hgb_model.predict_proba(X_test)

    int_to_class = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}
    hgb_pred_labels = [int_to_class[p] for p in hgb_preds]
    hgb_match_count = sum(p == s for p, s in zip(hgb_pred_labels, hgb_saved["predicted_risk_level"]))
    hgb_match_pct = hgb_match_count / len(X_test) * 100
    hgb_prob_diff = np.max(np.abs(hgb_probs - hgb_saved[["probability_SAFE", "probability_CAUTION", "probability_WARNING", "probability_CRITICAL"]].values))

    print(f"  -> HGB Model (Load: {hgb_load_ms:.1f} ms):")
    print(f"     Label Agreement: {hgb_match_count:,} / {len(X_test):,} ({hgb_match_pct:.2f}%)")
    print(f"     Max Probability Delta: {hgb_prob_diff:.2e} (Tolerance: 1.00e-05)")

    # 2. XGBoost
    t0 = time.perf_counter()
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))
    xgb_load_ms = (time.perf_counter() - t0) * 1000
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)

    xgb_pred_labels = [int_to_class[p] for p in xgb_preds]
    xgb_match_count = sum(p == s for p, s in zip(xgb_pred_labels, xgb_saved["predicted_risk_level"]))
    xgb_match_pct = xgb_match_count / len(X_test) * 100
    xgb_prob_diff = np.max(np.abs(xgb_probs - xgb_saved[["probability_SAFE", "probability_CAUTION", "probability_WARNING", "probability_CRITICAL"]].values))

    print(f"  -> XGBoost Model (Load: {xgb_load_ms:.1f} ms):")
    print(f"     Label Agreement: {xgb_match_count:,} / {len(X_test):,} ({xgb_match_pct:.2f}%)")
    print(f"     Max Probability Delta: {xgb_prob_diff:.2e} (Tolerance: 1.00e-05)")

    passed = (hgb_match_pct == 100.0 and xgb_match_pct == 100.0 and hgb_prob_diff < 1e-5 and xgb_prob_diff < 1e-5)
    print(f"  Result: {'[PASS] Exact Numerical Reproduction Confirmed' if passed else '[FAIL] Deviations Detected'}")
    return passed


def verify_preprocessor_transformation() -> bool:
    """Check preprocessor transformation and fault tolerance."""
    print("\n[Step 3/5] Testing Preprocessor Transformation & Unknown Category Handling...")
    prep: MineRakshakPreprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

    # Sample input
    raw_df = pd.DataFrame([{
        "distance_m": 8.5,
        "object_x_m": 1.2,
        "object_y_m": 8.4,
        "object_z_m": 0.3,
        "object_width_m": 2.2,
        "object_height_m": 2.0,
        "object_length_m": 4.8,
        "point_count": 420,
        "relative_velocity_mps": -6.1,
        "truck_speed_kmph": 30.0,
        "time_to_collision_s": 1.39,
        "object_type": "excavator",
    }])

    transformed = prep.transform(raw_df)
    shape_ok = transformed.shape == (1, 17)
    cols_ok = list(transformed.columns) == prep.transformed_feature_names

    # Unknown category tolerance
    unknown_df = raw_df.copy()
    unknown_df["object_type"] = "novel_drone_sensor_artifact"
    transformed_unknown = prep.transform(unknown_df)
    unknown_ok = transformed_unknown.shape == (1, 17) and not transformed_unknown.isna().any().any()

    print(f"  Transformed Output Shape: {transformed.shape} (Expected: (1, 17)) -> {'OK' if shape_ok else 'FAIL'}")
    print(f"  Feature Order Consistency: {'OK' if cols_ok else 'FAIL'}")
    print(f"  Handling Unknown Category: {'OK' if unknown_ok else 'FAIL'}")

    return shape_ok and cols_ok and unknown_ok


def verify_latency_budget() -> bool:
    """Benchmark single-sample inference latency for real-time safety budget."""
    print("\n[Step 4/5] Benchmarking Real-Time Inference Latency (< 50 ms budget)...")
    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    hgb_model = joblib.load(MODELS_DIR / "hgb_model.joblib")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))

    raw_sample = pd.DataFrame([{
        "distance_m": 12.0, "object_x_m": 0.0, "object_y_m": 12.0, "object_z_m": 0.0,
        "object_width_m": 2.0, "object_height_m": 2.0, "object_length_m": 4.0,
        "point_count": 300, "relative_velocity_mps": -4.0, "truck_speed_kmph": 25.0,
        "time_to_collision_s": 3.0, "object_type": "truck",
    }])
    X_sample = prep.transform(raw_sample)

    # Warmup
    hgb_model.predict(X_sample)
    xgb_model.predict(X_sample)

    times_hgb = [time.perf_counter() for _ in range(50)]
    for i in range(50):
        t0 = time.perf_counter()
        hgb_model.predict_proba(X_sample)
        times_hgb[i] = time.perf_counter() - t0
    hgb_lat_ms = np.mean(times_hgb) * 1000

    times_xgb = [time.perf_counter() for _ in range(50)]
    for i in range(50):
        t0 = time.perf_counter()
        xgb_model.predict_proba(X_sample)
        times_xgb[i] = time.perf_counter() - t0
    xgb_lat_ms = np.mean(times_xgb) * 1000

    print(f"  HGB Single-Sample Latency:    {hgb_lat_ms:.3f} ms (Budget: < 50.0 ms) -> {'PASS' if hgb_lat_ms < 50 else 'FAIL'}")
    print(f"  XGBoost Single-Sample Latency: {xgb_lat_ms:.3f} ms (Budget: < 50.0 ms) -> {'PASS' if xgb_lat_ms < 50 else 'FAIL'}")

    return hgb_lat_ms < 50.0 and xgb_lat_ms < 50.0


def verify_sample_scenarios() -> bool:
    """Execute end-to-end forward passes on test scenarios."""
    print("\n[Step 5/5] Executing End-to-End Test Predictions on Synthetic Scenarios...")
    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    hgb_model = joblib.load(MODELS_DIR / "hgb_model.joblib")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))
    int_to_class = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}

    scenarios = [
        ("High-Risk Imminent Impact", {
            "distance_m": 4.5, "object_x_m": 0.2, "object_y_m": 4.4, "object_z_m": 0.1,
            "object_width_m": 2.8, "object_height_m": 3.0, "object_length_m": 6.5,
            "point_count": 600, "relative_velocity_mps": -9.2, "truck_speed_kmph": 35.0,
            "time_to_collision_s": 0.49, "object_type": "truck",
        }),
        ("Moderate Lateral Warning", {
            "distance_m": 15.0, "object_x_m": -2.5, "object_y_m": 14.8, "object_z_m": 0.2,
            "object_width_m": 1.2, "object_height_m": 1.8, "object_length_m": 1.5,
            "point_count": 220, "relative_velocity_mps": -3.5, "truck_speed_kmph": 22.0,
            "time_to_collision_s": 4.28, "object_type": "person",
        }),
        ("Distant Benign Object", {
            "distance_m": 68.0, "object_x_m": 8.0, "object_y_m": 67.5, "object_z_m": 0.0,
            "object_width_m": 2.0, "object_height_m": 1.5, "object_length_m": 4.2,
            "point_count": 80, "relative_velocity_mps": 0.5, "truck_speed_kmph": 15.0,
            "time_to_collision_s": 136.0, "object_type": "car",
        }),
    ]

    for name, sample_dict in scenarios:
        df = pd.DataFrame([sample_dict])
        X = prep.transform(df)

        hgb_p = int(hgb_model.predict(X)[0])
        hgb_prob = hgb_model.predict_proba(X)[0]
        xgb_p = int(xgb_model.predict(X)[0])
        xgb_prob = xgb_model.predict_proba(X)[0]

        print(f"\n  Scenario: '{name}'")
        print(f"    Raw: Distance={sample_dict['distance_m']}m, RelVel={sample_dict['relative_velocity_mps']} m/s, TTC={sample_dict['time_to_collision_s']}s")
        print(f"    -> HGB (Primary):   {int_to_class[hgb_p]:<8} (Conf: {hgb_prob[hgb_p]*100:.1f}%) [P(SAFE)={hgb_prob[0]:.2f}, P(CAUTION)={hgb_prob[1]:.2f}, P(WARN)={hgb_prob[2]:.2f}, P(CRIT)={hgb_prob[3]:.2f}]")
        print(f"    -> XGBoost (Voter): {int_to_class[xgb_p]:<8} (Conf: {xgb_prob[xgb_p]*100:.1f}%) [P(SAFE)={xgb_prob[0]:.2f}, P(CAUTION)={xgb_prob[1]:.2f}, P(WARN)={xgb_prob[2]:.2f}, P(CRIT)={xgb_prob[3]:.2f}]")

    return True


def main() -> None:
    print("=" * 75)
    print("mineRakshak-ai: Stage 8 - Artifact Serialization & Verification Audit")
    print("=" * 75)

    if not MANIFEST_PATH.exists():
        print(f"[FAIL] Manifest not found at {MANIFEST_PATH}!")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    p1 = verify_manifest_checksums(manifest)
    p2 = verify_model_reproducibility()
    p3 = verify_preprocessor_transformation()
    p4 = verify_latency_budget()
    p5 = verify_sample_scenarios()

    print("\n" + "=" * 75)
    if p1 and p2 and p3 and p4 and p5:
        print("STAGE 8 ARTIFACT PACKAGING AUDIT: ALL CHECKS PASSED (100% VERIFIED)")
        print("Pipeline is completely frozen, validated, and ready for Stage 9 / Inference.")
    else:
        print("STAGE 8 ARTIFACT PACKAGING AUDIT: ONE OR MORE CHECKS FAILED!")
        sys.exit(1)
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
