# mineRakshak-ai: Stage 8 Artifact Packaging & Serialization Verification Report

> **SYNTHETIC PROTOTYPE NOTICE:**
> The serialized models, preprocessors, and manifests evaluated here are configured from synthetic tabular LiDAR and kinematic bounding box scenarios. They validate pipeline integrity, numerical reproducibility, and software contracts. They do **NOT** certify physical haul truck safety without real-world sensor calibration and hardware-in-the-loop (HIL) testing.

---

## 1. Executive Summary

Stage 8 finalizes the core artifact package for the **mineRakshak-ai** obstacle risk assessment engine. All models, transformers, schemas, mappings, and metadata have been frozen, version-tagged, cryptographically fingerprinted (SHA-256), and validated through an automated reproducibility test suite.

| Artifact Role | File Name | Format | Size | SHA-256 Checksum (First 16 chars) | Load Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Primary Production Model** | `hgb_model.joblib` | Joblib | 2.22 MB (2,327,920 B) | `f885c29d3ceddae8...` | Verified |
| **Secondary Benchmark Voter** | `xgboost_model.json` | JSON | 7.67 MB (8,040,228 B) | `bb385fb728583b38...` | Verified |
| **Frozen Preprocessor** | `preprocessor.joblib` | Joblib | 1.59 KB (1,629 B) | `37bc0e1366174965...` | Verified |
| **Feature Ordering Schema** | `feature_schema.json` | JSON | 947 B | `e9ca72a4b9a9675c...` | Verified |
| **Risk Label Mapping** | `label_mapping.json` | JSON | 319 B | `2b5fd687a4f99cc2...` | Verified |
| **Pipeline Manifest** | `manifest.json` / `model_manifest.json` | JSON | 3.54 KB | Linked Contract | Verified |
| **Model Metadata & Provenance** | `model_metadata.json` | JSON | 3.82 KB | Complete Lineage | Verified |

---

## 2. Frozen Artifact Manifest (`models/manifest.json`)

The manifest acts as the single source of truth for the inference pipeline. It defines:
1. **Model Roles**: `hgb_model.joblib` as `primary_production` (superior CRITICAL recall of `91.14%`, zero catastrophic errors, 3.45× smaller footprint, native scikit-learn runtime) and `xgboost_model.json` as `secondary_benchmark_voter` (`85.60%` test accuracy, `97.00%` prediction agreement).
2. **Preprocessor Contract**: `MineRakshakPreprocessor` fitted on 14,000 training samples with `handle_unknown="ignore"` to safeguard against unmodeled obstacle classes.
3. **Feature Schema**: 12 raw input features $\rightarrow$ 17 deterministic transformed features.
4. **Target Encoding**: `SAFE: 0`, `CAUTION: 1`, `WARNING: 2`, `CRITICAL: 3`.
5. **Runtime Environment**:
   * Python: `3.14.7`
   * Scikit-Learn: `1.9.1`
   * XGBoost: `3.4.1`
   * Joblib: `1.5.3`
   * NumPy: `2.4.2`
   * Pandas: `3.0.1`

---

## 3. Serialization & Numerical Reproducibility Audit

The automated verification suite ([tests/test_model_serialization.py](file:///c:/Users/ADMIN/Downloads/mineRakshak_backend/mineRakshak-ai/tests/test_model_serialization.py) and [src/inference/verify_artifacts.py](file:///c:/Users/ADMIN/Downloads/mineRakshak_backend/mineRakshak-ai/src/inference/verify_artifacts.py)) was executed against the untouched 3,000-sample test set.

### A. Checksum Integrity
* All 5 artifacts matched their recorded SHA-256 hashes with zero discrepancies.

### B. Prediction and Probability Reproduction
* **HistGradientBoostingClassifier (`hgb_model.joblib`)**:
  * Test Samples: `3,000`
  * Prediction Agreement vs Stage 7 Saved Predictions: **`3,000 / 3,000` (`100.00%`)**
  * Max Probability Deviation: **`1.11e-16`** (well below the $1.00\times 10^{-5}$ float threshold)
  * Cold Deserialization Latency: **`31.0 ms`**
* **XGBClassifier (`xgboost_model.json`)**:
  * Test Samples: `3,000`
  * Prediction Agreement vs Stage 7 Saved Predictions: **`3,000 / 3,000` (`100.00%`)**
  * Max Probability Deviation: **`2.98e-08`** (well below the $1.00\times 10^{-5}$ float threshold)
  * Cold Deserialization Latency: **`1,229.9 ms`**

### C. Preprocessor Robustness
* Transformed shape: exactly `(N, 17)`.
* Feature names and column ordering match `feature_schema.json`.
* Fault tolerance test: passing an unknown categorical class (`object_type="novel_drone_sensor_artifact"`) zero-fills all one-hot columns without raising exceptions or producing `NaN` / `Inf` values.

### D. Single-Sample Real-Time Latency Benchmark
* Haul truck perception deadline: **`< 50.0 ms`** (supporting 10–20 Hz publishing rates).
* **HGB Forward Pass**: **`8.99 ms`** ($\approx 111 \text{ Hz}$) $\rightarrow$ **PASS**
* **XGBoost Forward Pass**: **`2.27 ms`** ($\approx 440 \text{ Hz}$) $\rightarrow$ **PASS**

---

## 4. End-to-End Scenario Verification

To validate that the pipeline can consume raw dictionary inputs and return calibrated risk tiers, three representative mining scenarios were evaluated:

### Scenario 1: High-Risk Imminent Impact
* **Input**: Distance = `4.5 m`, RelVel = `-9.2 m/s`, TTC = `0.49 s`, Type = `truck`
* **Primary (HGB)**: **`CRITICAL`** (Confidence: `99.7%`) — $P(\text{CRIT}) = 1.00$
* **Voter (XGBoost)**: **`CRITICAL`** (Confidence: `99.5%`) — $P(\text{CRIT}) = 1.00$
* **Assessment**: Unanimous immediate emergency stop / full braking trigger.

### Scenario 2: Moderate Lateral Warning
* **Input**: Distance = `15.0 m`, RelVel = `-3.5 m/s`, TTC = `4.28 s`, Type = `person`
* **Primary (HGB)**: **`CAUTION`** (Confidence: `50.6%`) — $P(\text{CAUTION}) = 0.51$, $P(\text{SAFE}) = 0.48$
* **Voter (XGBoost)**: **`SAFE`** (Confidence: `72.0%`) — $P(\text{SAFE}) = 0.72$, $P(\text{CAUTION}) = 0.27$
* **Assessment**: Borderline kinematic threshold; correctly flags caution on vulnerable road user (pedestrian).

### Scenario 3: Distant Benign Object
* **Input**: Distance = `68.0 m`, RelVel = `+0.5 m/s` (receding), TTC = `136.0 s`, Type = `car`
* **Primary (HGB)**: **`SAFE`** (Confidence: `99.9%`) — $P(\text{SAFE}) = 1.00$
* **Voter (XGBoost)**: **`SAFE`** (Confidence: `99.8%`) — $P(\text{SAFE}) = 1.00$
* **Assessment**: Unanimous nominal clearance; zero false alarm.

---

## 5. Next Steps for Inference Engine (Stage 9)

With all artifacts frozen, fingerprinted, and verified:
1. Build `src/inference/predictor.py`: A high-performance inference engine wrapping the manifest, preprocessor, and primary/secondary models with automatic schema validation and dual-voter arbitration.
2. Build `src/inference/schemas.py`: Pydantic / dataclass input and output contracts for REST / ROS 2 message bridging.
3. Expose synchronous batch and single-sample prediction APIs ready for ROS 2 node integration.
