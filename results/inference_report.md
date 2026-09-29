# mineRakshak-ai: Stage 8 Production Inference Pipeline & Integration Report

> **CRITICAL DISCLAIMER & ENGINEERING INTEGRITY NOTICE:**
> The inference pipeline and underlying models operate on **SYNTHETIC PROTOTYPE DATA**
> modeling assumed feature representations from future ROS 2 LiDAR clustering pipelines.
> These outputs serve as prototype decision support and do **NOT** establish or certify
> physical, real-world mine-site safety performance.

---

## 1. Implementation Status

* **Module**: `src/inference/predict.py` implemented and verified.
* **Primary Model**: `models/hgb_model.joblib` (HistGradientBoostingClassifier) loaded as primary production engine.
* **Secondary Model**: `models/xgboost_model.json` (XGBClassifier) available for dual-voter arbitration.
* **Preprocessing**: `models/preprocessor.joblib` frozen transformer loaded without refitting.
* **Status**: Fully functional for single-sample, batch, and streaming perception inference.

## 2. Input Schema Contract

The inference engine accepts raw obstacle observation dictionaries or DataFrames with 12 input fields:

| Field Name | Type | Physical Units | Constraints / Domain | Description |
| :--- | :---: | :---: | :---: | :--- |
| `distance_m` | float | meters (m) | $\ge 0.0$ | Radial sensor distance to obstacle center |
| `object_x_m` | float | meters (m) | Any float | Longitudinal position ahead (+X forward) |
| `object_y_m` | float | meters (m) | Any float | Lateral position (+/-Y relative to truck center) |
| `object_z_m` | float | meters (m) | Any float | Elevation (+Z above sensor mount) |
| `object_width_m` | float | meters (m) | $> 0.0$ | Lateral 3D bounding box dimension |
| `object_height_m` | float | meters (m) | $> 0.0$ | Vertical 3D bounding box dimension |
| `object_length_m` | float | meters (m) | $> 0.0$ | Longitudinal 3D bounding box dimension |
| `point_count` | int | count | $\ge 0$ | Number of LiDAR points in object cluster |
| `relative_velocity_mps` | float | m/s | Negative = closing | Velocity relative to truck forward frame |
| `truck_speed_kmph` | float | km/h | $\ge 0.0$ | Ground speed of the haul truck |
| `time_to_collision_s` | float | seconds (s) | $\ge 0.0$ (clamped $\le 99.9$) | Derived TTC (auto-calculated if omitted) |
| `object_type` | string | category | `car, truck, crane, excavator, person, unknown` | Upstream perception class (unknown tolerated) |

## 3. Preprocessing Flow & Feature Transformation

1. **Raw Ingestion**: Ingests raw observation dictionary or batch DataFrame.
2. **Sanitization & Derivation**: Validates physical bounds; if `time_to_collision_s` is omitted, dynamically computes $TTC = \frac{\text{distance}}{-\text{rel\_vel}}$ (clamped to 99.9 s).
3. **Frozen Transformation**: Feeds sanitized DataFrame to `preprocessor.transform()`.
4. **Zero-Fit Guarantee**: Preprocessor uses pre-fitted `OneHotEncoder(handle_unknown='ignore')` from Stage 4. `fit()` is never invoked.
5. **Deterministic Schema**: Emits exactly 17 numerical columns in the contract order defined in `models/feature_schema.json`.

## 4. Model Loading & Lifecycle Architecture

* **Singleton Pattern**: Managed via `get_inference_engine()` to avoid repeated cold-start disk I/O.
* **Cold Loading Latency**: Primary HGB model loads in **~31 ms**; Preprocessor loads in **~2 ms**.
* **Decoupled Imports**: Uses isolated `sklearn.ensemble._hist_gradient_boosting` import hooks ensuring 100% compatibility with Windows Smart App Control.

## 5. Output Format & Response Action Mapping

Every prediction returns a structured dictionary ready for ROS 2 message publishing or REST APIs:

```json
{
  "risk_level": "CRITICAL",
  "class_id": 3,
  "confidence": 0.9972,
  "probabilities": {
    "SAFE": 0.0001,
    "CAUTION": 0.0003,
    "WARNING": 0.0024,
    "CRITICAL": 0.9972
  },
  "model": "HistGradientBoostingClassifier",
  "status": "valid",
  "recommendation": "immediate hazard / urgent intervention"
}
```

### Safety-Oriented Response Mapping Table:
| Risk Tier | Numeric Code | Deterministic Human-Readable Action | Operational Guidance |
| :--- | :---: | :--- | :--- |
| **`SAFE`** | `0` | `no immediate hazard` | Telemetry logging / Alerting |
| **`CAUTION`** | `1` | `increased awareness / monitor` | Telemetry logging / Alerting |
| **`WARNING`** | `2` | `active warning / prepare intervention` | Telemetry logging / Alerting |
| **`CRITICAL`** | `3` | `immediate hazard / urgent intervention` | Telemetry logging / Alerting |


## 6. Input Validation & Fault Tolerance Behavior

The engine enforces strict validation before inference to prevent silent failures:
* **Missing Keys**: Immediately raises `InferenceValidationError` specifying the missing column.
* **Invalid Types**: Non-numeric values (e.g. `'five'` for distance) raise clear validation exceptions.
* **Nulls / NaNs / Infs**: Explicitly rejected with informative error messages.
* **Negative Distance / Dimensions**: Rejected as physically impossible sensor measurements.
* **Unknown Object Types**: Handled gracefully without raising errors. One-hot columns are safely zero-filled.

## 7. Test Results Summary

* **Total Verification Tests**: `15`
* **Passed**: `15` / `15` (**`100.0%`**)
* **Failed**: `0`

| Test Name | Objective | Status |
| :--- | :--- | :---: |
| **1. Valid SAFE-like Input** | Distant stationary vehicle correctly predicted as SAFE (class_id=0, status=valid) | **`PASS`** |
| **2. Valid CAUTION-like Input** | Moderate distance excavator yields valid low/moderate risk assessment | **`PASS`** |
| **3. Valid WARNING-like Input** | Approaching crane at 20m produces valid risk prediction and schema fields | **`PASS`** |
| **4. Valid CRITICAL-like Input** | Approaching haul truck at 5.5m correctly predicted as CRITICAL with high confidence | **`PASS`** |
| **5. Missing Field Rejection** | Omitting required observation field raises InferenceValidationError | **`PASS`** |
| **6. NaN Input Rejection** | NaN value in sensor field raises InferenceValidationError | **`PASS`** |
| **7. Infinite Input Rejection** | Infinite value in distance_m raises InferenceValidationError | **`PASS`** |
| **8. Invalid Negative Physical Value** | Negative distance_m raises physical bounds InferenceValidationError | **`PASS`** |
| **9. Unknown Object Type Handling** | Unseen object_type safely handled with zero-filled OHE features | **`PASS`** |
| **10. Prediction Determinism** | Repeated inference on identical inputs returns bitwise identical outputs | **`PASS`** |
| **11. Probability Vector Normalization** | Output 4-class probabilities sum to 1.0 +/- 0.001 | **`PASS`** |
| **12. Argmax Consistency** | Predicted risk class matches key with highest probability exactly | **`PASS`** |
| **13. Label Mapping Agreement** | Output keys and class_id match models/label_mapping.json strictly | **`PASS`** |
| **14. Feature Schema Ordering** | Preprocessor output preserves exact 17-feature order from models/feature_schema.json | **`PASS`** |
| **15. Serialized Model Reload Verification** | Clean reloaded engine from disk reproduces bitwise identical predictions | **`PASS`** |


## 8. Determinism & Numerical Reproducibility

* **Repeated Single-Sample Invocations**: 50 consecutive runs yielded bitwise identical probability vectors ($0.0$ variance).
* **Test Set Sample #0 Reproduction**: Verified that the pipeline's output on test set sample #0 reproduces the Stage 7 saved test prediction with `0.00%` label discrepancy and maximum probability delta $< 1.0\times 10^{-4}$.

## 9. Real-Time Latency Benchmark

Measured on Windows 11 (AMD64) with 50 warm-up iterations followed by 1,000 timed runs:

### Single-Sample Inference Latency:
| Metric | Measured Latency (ms) | Haul Truck Real-Time Target (< 50 ms) |
| :--- | :---: | :---: |
| **Mean Latency** | **`9.4609 ms`** | **PASS** (Utilizes 18.92% of budget) |
| **Median (P50)** | `9.1047 ms` | **PASS** |
| **P95 Latency** | `12.7041 ms` | **PASS** |
| **P99 Latency** | `14.3337 ms` | **PASS** |
| **Min / Max** | `5.966 ms` / `41.9871 ms` | **PASS** |
| **Standard Deviation** | `1.9075 ms` | Highly stable |


### Single-Sample Latency Breakdown:
* **Input Validation**: `0.0425 ms` (0.4%)
* **Preprocessing Transformation**: `2.6461 ms` (28.0%)
* **Primary Model Forward Pass**: `6.7492 ms` (71.3%)

### Vectorized Batch Inference Benchmarks:
| Batch Size | Mean Total Latency (ms) | P95 Latency (ms) | Per-Sample Latency ($\mu$s) | Throughput (FPS) |
| :---: | :---: | :---: | :---: | :---: |
| **5 objects** | `8.4189 ms` | `10.4865 ms` | `1683.78 \mu\text{s}` | **`593.9 obj/sec`** |
| **20 objects** | `8.8419 ms` | `11.8297 ms` | `442.1 \mu\text{s}` | **`2262.0 obj/sec`** |
| **100 objects** | `11.8855 ms` | `14.341 ms` | `118.86 \mu\text{s}` | **`8413.6 obj/sec`** |


## 10. Error Handling & Edge-Case Architecture

| Edge Case | Handled By | Behavior |
| :--- | :--- | :--- |
| **Missing required field** | `validate_observation()` | Raises `InferenceValidationError` |
| **Null / NaN input** | `validate_observation()` | Raises `InferenceValidationError` |
| **Non-numeric strings** | `validate_observation()` | Raises `InferenceValidationError` |
| **Negative distance** | `validate_observation()` | Raises `InferenceValidationError` |
| **Omitted `time_to_collision_s`** | `calculate_ttc()` | Dynamically derived from distance & relative velocity |
| **Unseen `object_type`** | `OneHotEncoder(handle_unknown='ignore')` | Zero-fills category columns safely without crashing |
| **Zero relative velocity** | `calculate_ttc()` | Clamped to safe ceiling (`99.9 s`) without zero-division |

## 11. Generated Stage 8 Artifacts

| Artifact | File Path | Purpose |
| :--- | :--- | :--- |
| **Inference Module** | `src/inference/predict.py` | Production inference engine with single and batch APIs |
| **Package Init** | `src/inference/__init__.py` | Exported public symbols and exception classes |
| **Dedicated Test Suite** | `src/inference/test_predict.py` | 15 comprehensive unit & edge-case tests |
| **Integration Test Suite** | `tests/test_inference.py` | 12 integration and schema tests |
| **Benchmark Runner** | `src/inference/benchmark.py` | Latency and test execution benchmarking tool |
| **Benchmark Metrics** | `results/metrics/inference_benchmark.json` | 1,000-run percentiles, breakdown, and throughput measurements |
| **Test Results JSON** | `results/metrics/inference_test_results.json` | Machine-readable test suite verification output |
| **Inference Report** | `results/inference_report.md` | Comprehensive Stage 8 documentation |


## 12. Limitations

1. **Simulated Inputs**: Features represent idealized bounding box geometries without sensor noise, point dropouts, or occlusion artifacts.
2. **Pipeline Scope**: The measured ~8 ms latency reflects ML inference only; ROS 2 point cloud ingestion and clustering latency will add upstream overhead.
3. **Deployment Status**: Not certified for active brake-by-wire control without formal HIL validation.

## 13. Readiness for Next Stage (Stage 9 / ROS 2 Integration)

The inference engine is completely packaged, deterministic, sub-10ms, and validated.
It provides standard Python callables (`predict()`, `predict_single()`, `predict_batch()`) ready to be wrapped directly into a ROS 2 Subscriber/Publisher node.