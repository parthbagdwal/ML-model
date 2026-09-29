# Stage 10: End-to-End MineRakshak System Validation and Demonstration Report

**Document ID:** MR-VAL-STG10-001  
**Project:** MineRakshak AI — Mining Haul Truck Safety Risk Prediction Engine  
**Stage:** 10 — System-Level Validation, Safety Boundaries, and Latency Benchmarking  
**Primary Production Candidate:** `HistGradientBoostingClassifier` (`models/hgb_model.joblib`)  
**Secondary Reference Model:** `XGBClassifier` (`models/xgboost_model.json`)  
**Preprocessing Artifact:** `MineRakshakPreprocessor` (`models/preprocessor.joblib`)  
**Feature Ordering:** `models/feature_schema.json` (17 transformed features)  
**Label Mapping:** `models/label_mapping.json` (`SAFE: 0`, `CAUTION: 1`, `WARNING: 2`, `CRITICAL: 3`)  
**Execution Timestamp:** 2026-09-28T19:35:40Z  
**Verdict:** **STAGE 10 SYSTEM VALIDATION PASSED (56/56 Tests Passing, 8/8 Scenarios Verified, Latency Budget Compliant)**

---

> [!CAUTION]
> ### Critical Prototype Decision-Support Disclaimer
> **This software is a research prototype decision-support engine developed and evaluated using synthetic sensor features.**  
> It has **NOT** been certified or tested on physical haul trucks operating within active mine sites.  
> The system **CANNOT and DOES NOT** actuate vehicle brakes, steer the vehicle, alter throttle, or control mechanical equipment. All outputs are advisory situational risk evaluations intended for driver assistance and safety telemetry monitoring.

---

## 1. Executive Summary

Stage 10 represents the final system-level validation milestone for the MineRakshak AI perception-to-risk prediction pipeline prior to operational field integration planning. In strict compliance with architectural constraints:
1. **Zero Model Retraining / Refitting**: The primary `HistGradientBoostingClassifier` (`models/hgb_model.joblib`), secondary `XGBoost` model, and `preprocessor.joblib` were loaded completely frozen.
2. **Deterministic Demonstration Pipeline**: Built an end-to-end evaluation pipeline that ingests simulated LiDAR obstacle observations, applies strict physical validation, transforms 12 raw attributes through the frozen 17-feature preprocessor, infers 4-class risk probabilities, generates situational advisories, isolates highest-threat obstacles, and packages outputs into standard ROS 2 payloads.
3. **8 Representative Scenarios Evaluated**: Verified Scenarios A through H covering all four risk classes (`SAFE`, `CAUTION`, `WARNING`, `CRITICAL`), mixed multi-obstacle frames, malformed sensor fault isolation, unknown obstacle categories, and multi-threat resolution.
4. **Safety Boundaries Enforced**: Verified through automated tests that `CRITICAL` threats are never downgraded, `WARNING` threats are never downgraded to `SAFE`, invalid sensor inputs never yield a `valid` prediction, and outputs strictly lack vehicle actuation fields.
5. **Real-Time Latency Conformance**: Benchmarked latency across frames containing 1, 5, 10, 20, 50, and 100 detected obstacles. At 100 objects, P95 frame processing latency was **20.26 ms**, consuming only **40.5%** of the 50.0 ms real-time cycle budget.
6. **Complete Test Suite**: Executed 56 unit and integration tests across 4 test suites with **100% pass rate** (0 failures, 0 errors, 0 skips).

---

## 2. Stages 1–9 Artifact Integrity Verification

A full repository audit confirmed that all predecessor artifacts from Stages 1 through 9 remain intact, unmodified, and internally consistent:

| Stage | Artifact Description | Location | Status | Checksum / Fingerprint |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 & 2** | Synthetic Dataset & Analysis | `data/raw/synthetic_mine_data.csv` | Intact | 20,000 samples, 13 raw columns |
| **Stage 3** | Feature Schema Contract | `models/feature_schema.json` | Intact | 12 raw features, 17 transformed features |
| **Stage 4** | Frozen Preprocessing Pipeline | `models/preprocessor.joblib` | Intact | Standard scalers + OneHotEncoder |
| **Stage 4** | Train / Val / Test Splits | `data/processed/` | Intact | 14,000 train / 3,000 val / 3,000 test |
| **Stage 5** | Baseline XGBoost Model | `models/xgboost_model.json` | Intact | Macro F1 = 0.9847, 8.04 MB |
| **Stage 6** | Primary HGB Production Model | `models/hgb_model.joblib` | Intact | Macro F1 = 0.9868, 2.33 MB |
| **Stage 7** | Model Evaluation & Selection | `results/model_comparison_report.md` | Complete | HGB selected: zero critical FN |
| **Stage 8** | Production Inference Layer | `src/inference/predict.py` | Verified | `MineRakshakInferenceEngine` |
| **Stage 8** | Model Metadata & Manifest | `models/manifest.json` | Verified | Artifact provenance & hash registry |
| **Stage 9** | ROS 2 Perception-to-Risk Node | `ros2_ws/src/minerakshak_risk/` | Verified | Subscribed: `/minerakshak/object_observations` |

---

## 3. End-to-End Pipeline Architecture

The end-to-end demonstration pipeline processes obstacle detections through nine deterministic stages:

```
[LiDAR / Perception Layer]
      │
      ▼
1. Raw Obstacle Cluster Ingestion (Single Dict or PerceptionFrame JSON)
      │
      ▼
2. Physical Input Validation (Rejects NaN, Inf, Negative Distances/Dimensions)
      │
      ├──> Fault Detected ──> Trapped & Isolated: status="error", risk_level="INVALID" (No Node Crash)
      │
      ▼ (Valid Input)
3. Feature Derivation & Preprocessing (Frozen preprocessor.joblib: 17 Features)
      │
      ▼
4. Model Inference (Frozen HistGradientBoostingClassifier)
      │
      ▼
5. Risk Classification (SAFE=0, CAUTION=1, WARNING=2, CRITICAL=3)
      │
      ▼
6. Calibrated Probability & Confidence Extraction (sum == 1.0, argmax check)
      │
      ▼
7. Situational Recommendation Assignment (Advisory human-readable action)
      │
      ▼
8. Multi-Obstacle Frame Evaluation & Highest-Threat Selection (CRITICAL > WARNING > CAUTION > SAFE)
      │
      ▼
9. ROS 2 Structured Output Publishing:
      ├── /minerakshak/risk_prediction (Detailed individual evaluations)
      └── /minerakshak/highest_threat (Cycle summary & top hazard alert)
```

---

## 4. Deterministic Demonstration Scenarios (A through H)

Eight representative operational scenarios were evaluated using the frozen pipeline. All scenarios executed with bitwise reproducibility and zero runtime exceptions.

### Scenario Descriptions & Results Table

| Scenario ID | Name & Description | Expected Risk | Predicted Risk | Class ID | Confidence | P(SAFE) | P(CAUT) | P(WARN) | P(CRIT) | Status | Latency | Verdict |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario A** | SAFE: Distant Pedestrian (36.2m, receding) | `SAFE` | `SAFE` | 0 | 0.9701 | 0.9701 | 0.0295 | 0.0004 | 0.0000 | valid | 25.21 ms | **PASS** |
| **Scenario B** | CAUTION: Lateral Car (12.8m, lateral corridor) | `CAUTION` | `CAUTION` | 1 | 0.9663 | 0.0274 | 0.9663 | 0.0044 | 0.0019 | valid | 9.57 ms | **PASS** |
| **Scenario C** | WARNING: Pedestrian in Lane (15.7m, closing) | `WARNING` | `WARNING` | 2 | 0.9212 | 0.0049 | 0.0612 | 0.9212 | 0.0126 | valid | 9.02 ms | **PASS** |
| **Scenario D** | CRITICAL: Head-on Imminent Car (6.3m, -14.5 m/s) | `CRITICAL` | `CRITICAL` | 3 | 0.9997 | 0.0000 | 0.0000 | 0.0002 | 0.9997 | valid | 8.11 ms | **PASS** |
| **Scenario E** | Mixed Multi-Obstacle Frame (A, B, C, D in 1 frame) | `CRITICAL` | `CRITICAL` | 3 | 1.0000 | — | — | — | — | valid | 8.02 ms | **PASS** |
| **Scenario F** | Malformed Sensor Input (-15m distance, -2m width) | `INVALID` | `INVALID` | -1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | error | 0.16 ms | **PASS** |
| **Scenario G** | Unknown Object Type (`inspection_quadcopter`) | `SAFE` | `SAFE` | 0 | 0.9825 | 0.9825 | 0.0172 | 0.0002 | 0.0000 | valid | 9.16 ms | **PASS** |
| **Scenario H** | Simultaneous Threats (WARNING + CRITICAL in frame) | `CRITICAL` | `CRITICAL` | 3 | 1.0000 | — | — | — | — | valid | 8.35 ms | **PASS** |

### Scenario Deep Dive

1. **Scenario A (SAFE Obstacle)**:
   - *Physical Parameters*: Distance $36.2\text{ m}$, relative velocity $+13.45\text{ m/s}$ (receding), TTC $99.9\text{ s}$, object type `person`.
   - *Outcome*: Correctly classified as `SAFE` with $97.01\%$ confidence. Probabilities: SAFE: 0.9701, CAUTION: 0.0295, WARNING: 0.0004, CRITICAL: 0.0000. Sum $= 1.0000$.
   - *Recommendation*: `"no immediate hazard"`.

2. **Scenario B (CAUTION Obstacle)**:
   - *Physical Parameters*: Distance $12.8\text{ m}$, lateral displacement $Y = -12.36\text{ m}$ (outside travel corridor), TTC $1.01\text{ s}$, object type `car`.
   - *Outcome*: Correctly classified as `CAUTION` with $96.63\%$ confidence. Lateral clearance prevents premature escalation to CRITICAL while flagging proximity.
   - *Recommendation*: `"increased awareness / monitor"`.

3. **Scenario C (WARNING Obstacle)**:
   - *Physical Parameters*: Distance $15.68\text{ m}$, lateral displacement $Y = -2.05\text{ m}$ (inside truck forward swath), relative velocity $-0.68\text{ m/s}$, object type `person`.
   - *Outcome*: Correctly classified as `WARNING` with $92.12\%$ confidence. Vulnerable road user in path prompts heightened threat evaluation.
   - *Recommendation*: `"active warning / prepare intervention"`.

4. **Scenario D (CRITICAL Obstacle)**:
   - *Physical Parameters*: Distance $6.30\text{ m}$, forward position $X = 5.87\text{ m}$, closing velocity $-14.45\text{ m/s}$ (approx. $52\text{ km/h}$ closing rate), TTC $0.44\text{ s}$, object type `car`.
   - *Outcome*: Classified as `CRITICAL` with $99.97\%$ confidence. Probability for CRITICAL $= 0.9997$, near certainty.
   - *Recommendation*: `"immediate hazard / urgent intervention"`.

5. **Scenario E (Mixed Multi-Obstacle Frame)**:
   - *Composition*: 4 detected objects (A, B, C, and D) in a single LiDAR perception frame.
   - *Outcome*: All 4 objects evaluated in a single vector batch ($8.02\text{ ms}$ total cycle time). Individual risk levels preserved. The highest threat selector isolated object `obs_E_critical` with threat level `CRITICAL`.
   - *Recommendation*: `"immediate hazard / urgent intervention"`.

6. **Scenario F (Invalid Sensor Input / Malformed Obstacle)**:
   - *Anomaly*: Corrupt range reading (`distance_m = -15.0`), negative dimension (`object_width_m = -2.0`), negative point count (`-10`).
   - *Outcome*: Physical validator rejected observation before model inference. Trapped error message: `"'distance_m' cannot be negative: -15.0"`. Return schema populated with `status: "error"`, `risk_level: "INVALID"`, `class_id: -1`. The node remained fully operational with zero unhandled exceptions.

7. **Scenario G (Unknown Object Type)**:
   - *Anomaly*: Novel categorical label `autonomous_inspection_quadcopter` absent from training dictionary (`car, truck, crane, excavator, person, unknown`).
   - *Outcome*: One-hot encoding safely defaulted to an all-zero vector for categorical channels. Numerical kinematics dominated evaluation; classified safely as `SAFE` ($98.25\%$ confidence), proving zero-crash robustness on out-of-vocabulary sensor outputs.

8. **Scenario H (Multiple Simultaneous Threats)**:
   - *Composition*: Perception frame containing both a `WARNING` obstacle (`obs_H_warning_object`) and a `CRITICAL` obstacle (`obs_H_critical_object`).
   - *Outcome*: Highest-threat priority logic (`CRITICAL: 3 > WARNING: 2 > CAUTION: 1 > SAFE: 0`) decisively selected `obs_H_critical_object`. Zero threat masking or down-ranking occurred.

---

## 5. Deterministic Behavior Audit

To verify deterministic predictability, Scenarios A, B, C, D, and G were subjected to **50 consecutive evaluation cycles**:
- **Probability Consistency**: Max deviation across iterations was $0.0000000000$ (bitwise identical outputs).
- **Class ID Invariance**: No label flipping occurred across any repetition.
- **Audit Verdict**: **100% DETERMINISTIC PASS** across all tested scenarios.

---

## 6. Real-Time Latency Benchmark & Perception Budget

The pipeline was profiled across frames containing 1 to 100 simultaneous detected obstacles using 200 measured iterations per count after 25 warmup iterations.

### Latency Percentiles vs. Budget Table

| Obstacles per Frame | Mean Latency (ms) | Median P50 (ms) | P90 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Max Latency (ms) | Std Dev (ms) | Budget Limit (ms) | Budget Util. (%) | Headroom (ms) | Throughput (obj/s) | Conformance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 9.642 | 9.347 | 11.831 | **12.766** | 13.692 | 14.804 | 1.435 | 50.0 | 25.5% | +37.234 | 103.7 | **PASS** |
| **5** | 9.460 | 9.269 | 11.162 | **12.151** | 13.061 | 13.304 | 1.215 | 50.0 | 24.3% | +37.849 | 528.6 | **PASS** |
| **10** | 10.199 | 9.802 | 12.546 | **13.403** | 18.132 | 20.829 | 1.892 | 50.0 | 26.8% | +36.597 | 980.5 | **PASS** |
| **20** | 10.778 | 10.598 | 13.070 | **13.493** | 14.304 | 14.895 | 1.456 | 50.0 | 27.0% | +36.507 | 1,855.6 | **PASS** |
| **50** | 12.378 | 12.123 | 14.291 | **14.855** | 16.714 | 25.944 | 1.734 | 50.0 | 29.7% | +35.145 | 4,039.5 | **PASS** |
| **100** | 15.969 | 15.272 | 18.496 | **20.263** | 29.332 | 37.684 | 3.123 | 50.0 | **40.5%** | **+29.737** | **6,262.1** | **PASS** |

### Latency Scaling Visualization

The latency scaling profile against the $50\text{ ms}$ real-time budget is rendered below:

![MineRakshak AI Stage 10 Latency Scaling Plot](file:///c:/Users/ADMIN/Downloads/mineRakshak_backend/mineRakshak-ai/results/plots/stage10_e2e_latency.png)

### Performance Takeaways
1. **Comfortable Budget Headroom**: At 100 simultaneous objects (an extremely dense mining intersection), P95 latency is **20.26 ms**, leaving nearly **30 ms of headroom** for upstream LiDAR point cloud ground filtering and clustering.
2. **Batch Efficiency**: Vectorized feature extraction allows 100 objects to be evaluated in just $15.97\text{ ms}$ on average, yielding a peak inference throughput of over **6,260 objects/second**.
3. **P99 & Max Bound**: The absolute worst-case frame evaluation across 1,200 benchmark runs was $37.68\text{ ms}$ (at 100 objects), well below the $50.0\text{ ms}$ limit.

---

## 7. Safety Boundary Verification

Four non-negotiable safety rules were tested across combinatorial configurations in `tests/test_stage10_safety_boundaries.py`:

| Safety Boundary Rule | Test Method | Outcome | Status |
| :--- | :--- | :--- | :---: |
| **Rule 1: CRITICAL Never Downgraded** | Evaluated 4 permutations placing CRITICAL in first, middle, and last positions alongside SAFE, CAUTION, and WARNING obstacles. | Highest threat remained `CRITICAL` in 100% of trials. Zero downgrades to SAFE or CAUTION. | **PASS** |
| **Rule 2: WARNING Never Downgraded to SAFE** | Evaluated permutations with WARNING alongside single and multiple SAFE obstacles. | Highest threat was `WARNING` in 100% of trials. Never downgraded to SAFE. | **PASS** |
| **Rule 3: Invalid Sensor Data Fault Isolation** | Tested invalid inputs: negative distance, negative point count, negative width, NaN truck speed, infinite distance, None distance. | All 6 invalid conditions returned `status="error"`, `risk_level="INVALID"`, `class_id=-1`. None produced `valid`. | **PASS** |
| **Rule 4: Zero Vehicle Actuation Commands** | Scanned all output payloads, message schemas, and JSON dictionaries for 14 actuation commands (`brake`, `steering`, `throttle`, etc.). | Zero actuation fields detected. Outputs strictly contain risk level, class ID, probabilities, confidence, and situational text advice. | **PASS** |

---

## 8. ROS 2 Interface Conformance

The ROS 2 integration package (`ros2_ws/src/minerakshak_risk`) was verified against standard ROS 2 conventions:

### Topics & Schemas

| Topic Name | Message Type | Direction | Content |
| :--- | :--- | :---: | :--- |
| `/minerakshak/object_observations` | `std_msgs/String` (JSON) | Subscribed | Single obstacle observation or full `PerceptionFrame` |
| `/minerakshak/risk_prediction` | `std_msgs/String` (JSON) | Published | Complete risk assessment frame or single object prediction |
| `/minerakshak/highest_threat` | `std_msgs/String` (JSON) | Published | Immediate cycle hazard summary: `frame_id`, `highest_threat_level`, `action` |

### Parameter Configuration (`config/risk_node_params.yaml`)
- `input_topic`: `"/minerakshak/object_observations"`
- `output_topic`: `"/minerakshak/risk_prediction"`
- `threat_topic`: `"/minerakshak/highest_threat"`
- `model_path`: `"models/hgb_model.joblib"`
- `preprocessor_path`: `"models/preprocessor.joblib"`
- `feature_schema_path`: `"models/feature_schema.json"`
- `label_mapping_path`: `"models/label_mapping.json"`
- `enable_secondary_voter`: `false`
- `log_latency`: `true`

### Launch Configuration (`launch/risk_node.launch.py`)
- Configured to launch `minerakshak_risk_node` with parameters pre-mapped to production model paths and default topics.

---

## 9. Automated Test Suite Execution Results

The complete automated test suite was executed using Python's standard `unittest` framework:

```
Command: & ".\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

### Test Count Breakdown

| Test Suite File | Focus Area | Tests Run | Passed | Failed | Errors |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests/test_inference.py` | Stage 8 Production Inference Engine | 12 | 12 | 0 | 0 |
| `tests/test_model_serialization.py` | Stage 8 Artifact Persistence & Schemas | 11 | 11 | 0 | 0 |
| `tests/test_ros2_integration.py` | Stage 9 ROS 2 Perception Node & Frames | 18 | 18 | 0 | 0 |
| `tests/test_stage10_safety_boundaries.py` | Stage 10 System Scenarios & Safety Invariants | 15 | 15 | 0 | 0 |
| **Total Automated Tests** | **Complete MineRakshak Verification Suite** | **56** | **56** | **0** | **0** |

**Execution Time:** 7.743 seconds  
**Pass Rate:** **100.0%** (56/56 passing)

---

## 10. Operational Guidelines & Commands

### How to Run the End-to-End Demonstration
From the `mineRakshak-ai/` directory:
```powershell
& ".\.venv\Scripts\python.exe" -u "src/demo_stage10.py"
```

### How to Run the Latency Benchmark & Plot Generator
```powershell
& ".\.venv\Scripts\python.exe" -u "src/benchmark_stage10.py"
```

### How to Run the Full Automated Test Suite
```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

### How to Launch the ROS 2 Node (Inside a ROS 2 Workspace)
```bash
# 1. Source ROS 2 environment
source /opt/ros/humble/setup.bash

# 2. Build the package
colcon build --packages-select minerakshak_risk
source install/setup.bash

# 3. Launch via ROS 2 launch file
ros2 launch minerakshak_risk risk_node.launch.py

# Or run node executable directly
ros2 run minerakshak_risk risk_node
```

---

## 11. Final Stage 10 Verdict & Recommendations

### Final Verdict: **STAGE 10 VALIDATION PASSED**

1. **System Completeness**: The pipeline successfully executes end-to-end ingestion, schema verification, preprocessing, inference, threat prioritization, and output generation.
2. **Zero Failures**: All 56 automated tests passed without regressions or warning suppressions.
3. **Safety Assurance**: Safety boundaries (no down-ranking of critical hazards, full fault isolation on bad sensor inputs, zero vehicle actuation) are strictly enforced and verified.
4. **Latency Headroom**: Operates at $<41\%$ of the $50\text{ ms}$ real-time budget at maximum load (100 objects).

### Recommendations for Future Stages
* **Stage 11 Integration (Subject to Review & Approval)**:
  - Transition from simulated obstacle arrays to upstream point cloud clustering (e.g., ground plane removal + Euclidean clustering) via ROS 2 point cloud subscribers.
  - Integrate telemetry bridge to feed risk levels and situational advisories into the FastAPI dashboard backend.
  - Implement time-series tracking filter (e.g., Kalman filter / SORT) to track object state histories before passing features to the static tabular inference model.
* **Preserve Invariants**:
  - Maintain the frozen model weights (`hgb_model.joblib`) and preprocessor (`preprocessor.joblib`).
  - Retain strict zero-actuation safety boundary: MineRakshak remains an advisory situational awareness platform.

---
*Report certified by MineRakshak AI System Validation Team on 2026-09-28.*
