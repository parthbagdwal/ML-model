# mineRakshak-ai: Mining Truck Collision & Risk Prediction ML Pipeline

[![Status: Stage 15 Validated & Integration-Ready](https://img.shields.io/badge/Status-Stage%2015%20Validated%20%26%20Ready-brightgreen.svg)]()
[![Tests: 189/189 Passed](https://img.shields.io/badge/Tests-189%2F189%20Passing-success.svg)]()
[![Model: HistGradientBoosting (Frozen)](https://img.shields.io/badge/Primary%20Model-HGB%20Classifier%20(Frozen)-blue.svg)]()
[![10k Cycle Sustained Mean: 39.96ms](https://img.shields.io/badge/10k%20Sustained%20Mean-39.96ms%20%2F%2050ms-informational.svg)]()

---

> [!CAUTION]
> ### Crucial Operational & Hardware Safety Disclaimer
> 1. **Software Validated, Physical Hardware Validation Pending**: Stage 15 validates the entire software pipeline (10,000 sustained cycles, 40 dedicated tests, 189 regression tests), but physical 2D LiDAR hardware is not currently connected in this development environment. Physical validation remains required prior to field deployment.
> 2. **Synthetic vs. Real Data Provenance**: Synthetic test data is strictly segregated from real recorded streams. Synthetic data must never be reported or treated as physical hardware validation.
> 3. **Zero 3D Fabrication**: Missing 3D features (elevation $z$, vertical height, 3D volumetric bounding boxes, single-scan object classes) are **NOT fabricated**. They are explicitly flagged with transparent audit metadata.
> 4. **Decision Support & Alerting Only (Zero Actuation)**: The system is strictly an **advisory decision-support and driver-alerting system**. It **CANNOT and DOES NOT** actuate vehicle brakes, steer the vehicle, alter throttle, or command mechanical actuators.
> 5. **Frozen ML Artifacts**: All ML models, preprocessors, feature schemas, and label mappings remain byte-for-byte identical and frozen.

---

## 1. Project Overview & Current Status (Through Stage 15)

**mineRakshak-ai** is the AI/ML safety and risk assessment engine designed for haul trucks and heavy equipment operating in surface and underground mines.

In active mining environments, haul trucks face hazardous conditions including dense dust, heavy fog, blind spots, and close-proximity interactions with smaller support vehicles, cranes, excavators, and personnel. 

The goal of this ML component is to evaluate structured spatial and kinematic features of detected nearby objects along with the haul truck's operational state to predict a real-time risk level:
* **`SAFE` (0)** — Clear travel corridor; no immediate hazard.
* **`CAUTION` (1)** — Obstacle in vicinity or lateral zone; increased awareness required.
* **`WARNING` (2)** — Obstacle closing on path; active warning issued to operator.
* **`CRITICAL` (3)** — Imminent collision trajectory; urgent operator intervention required.

### Development Status Summary
* **Stages 1–7**: Formulated 17-feature schema, generated 20,000 synthetic training observations, trained & audited models, selecting **HistGradientBoostingClassifier** as primary production candidate (0.9868 Macro F1, 100.0% critical recall).
* **Stage 8**: Production inference engine with schema bounds validation and frozen artifact manifest.
* **Stage 9**: ROS 2 package `minerakshak_risk` with frame-level threat prioritization.
* **Stage 10**: Validated full pipeline across 8 operational scenarios, safety boundaries, and sub-50 ms latency.
* **Stage 11**: End-to-end full system integration: perception → ROS 2 `/minerakshak/object_observations` → frozen HGB → ROS 2 `/minerakshak/risk_prediction` + `/minerakshak/highest_threat` → FastAPI backend → WebSocket feeds (`/ws/dashboard`, `/ws/minerakshak/risk`).
* **Stage 12**: 2D planar LiDAR perception pipeline: `LaserScan2D`, `Obstacle2D`, clustering, multi-scan tracking (`MultiScanTracker2D`), and auditable feature mapping (`FeatureMapper2D`). Isolated legacy 3D adapter for regression.
* **Stage 13**: Real 2D LiDAR ROS 2 interface (`sensor_msgs/msg/LaserScan`), `ROS2LaserScanAdapter`, decoupled `SensorHealthState` machine, stale-scan watchdog (0.5s), and deterministic JSONL recording/replay.
* **Stage 14**: Configurable sensor profiles (`LiDAR2DConfig`), Cartesian calibration (`LiDARCalibration2D`), 16-metric scan quality diagnostics (`ScanQualityAnalyzer2D`), feature compatibility introspection, and offline dataset validation CLI.
* **Stage 15 (End-to-End System Validation, Fault Recovery & Deployment Readiness)**:
  - Complete architecture audit: 2D LaserScan input → adapter → calibration → clustering → tracking → feature mapping → frozen HGB inference → highest-threat selection → ROS 2 outputs → FastAPI state → WebSocket broadcast.
  - Resolved provenance forwarding and added 0.5s stale watchdog to FastAPI cache.
  - 2D sensor boundary audit confirmed **ZERO Category E (unintended runtime 3D processing)** in active pipeline.
  - 12-feature machine-readable provenance audit (`results/metrics/stage15_feature_provenance.json`).
  - Strict threat hierarchy enforcement: `CRITICAL > WARNING > CAUTION > SAFE`.
  - Software fault-injection suite: 17 fault conditions verified fail-safe with 100% clean recovery.
  - Multi-scan kinematic tracking validated: single scan invariant ($v_{\text{rel}} = 0.0$ m/s), closing velocity sign convention ($< 0$), and TTC derivation.
  - **Sustained Load Benchmark (10,000 Cycles)**: Mean software latency **39.964 ms** (vs. 50 ms budget), P95 **48.319 ms**, throughput **24.9 FPS**, memory growth **+0.38 MB**.
  - **189/189 automated tests passing** (40 dedicated Stage 15 tests + 149 regression tests).
  - All 5 frozen model artifact SHA-256 hashes verified bitwise identical.
  - 19-point physical validation readiness checklist established for future hardware integration.

---

## 2. Selected Production Model Candidate (Strictly Frozen)

* **Primary Production Model**: `HistGradientBoostingClassifier` (`models/hgb_model.joblib`, 2.33 MB, SHA-256: `f885c29d3ceddae8d2f5d7a2c1ff44fac849ec57ed498504938d632adb7f1974`)
* **Secondary / Reference Model**: `XGBClassifier` (`models/xgboost_model.json`, 8.04 MB, SHA-256: `bb385fb728583b385acbd1e7a052a4579a951835473dc10b12f6dccfdfacb983`)
* **Preprocessing Transformer**: `MineRakshakPreprocessor` (`models/preprocessor.joblib`, SHA-256: `37bc0e1366174965e741ab223ff3619c2e6fdc62e139b24ba5e8eaff3f0948da`)
* **Feature Schema**: `models/feature_schema.json` (SHA-256: `e9ca72a4b9a9675c8e85ad2059f0448f2140891512ad4d19c10280b90d2f986b`)
* **Label Mapping**: `models/label_mapping.json` (SHA-256: `2b5fd687a4f99cc2fcc94e3bd031c48571615193e7cff4de621a379492b17ebc`)

---

## 3. Real 2D LiDAR Ingestion Architecture (Stage 13)

```
[Real 2D Planar LiDAR Sensor / rosbag2 Replay]
                 │
                 ▼ (ROS 2 /scan: sensor_msgs/msg/LaserScan)
[ROS2LaserScanAdapter (src/perception/ros2_laserscan_adapter.py)]
  ├── Validate geometry (angle_min, angle_max, angle_increment > 0)
  ├── Validate bounds (range_min, range_max, range_min < range_max)
  ├── Filter individual beams (NaN, Inf, below min, above max)
  ├── Check timestamp freshness (stale timeout = 0.5s)
  ├── Maintain SensorHealthState (OK, NO_DATA, STALE, INVALID, DEGRADED)
  └── Apply mounting offsets (yaw rotation, x/y translation)
                 │
                 ▼ (LaserScan2D normalized planar scan)
[2D Obstacle Extraction & Segmentation (ObstacleExtractor2D)]
  ├── Polar-to-Cartesian (+X forward, +Y lateral left, Z not measured)
  └── Euclidean Jump-Distance Clustering (D_thresh = 1.2m, min_points = 2)
                 │
                 ▼ (list[Obstacle2D])
[Multi-Scan Temporal Tracker (MultiScanTracker2D)]
  ├── Gated nearest-neighbor centroid association (Gate = 4.5m)
  ├── Delta-Distance / Delta-Time closing velocity derivation
  └── Kinematic Time-to-Collision (TTC)
                 │
                 ▼ (Tracked Obstacles)
[Feature Compatibility & Mapping Layer (FeatureMapper2D)]
  ├── Direct 2D observations: distance, x, y, span width, depth, beam count
  ├── Kinematic velocity & TTC + external truck speed odometry
  ├── Nominal contract baselines: z = 0.0m, height = 1.5m, type = 'unknown'
  └── Transparent metadata audit: "MODEL COMPATIBILITY REQUIRES REAL-SENSOR VALIDATION"
                 │
                 ▼ (PerceptionFrame matching frozen 12-feature schema)
[Frozen HistGradientBoostingClassifier Engine]
                 │
                 ▼ (RiskAssessmentFrame)
[ROS 2 Node / FastAPI Backend / WebSocket Dashboard]
  ├── ROS 2 Topics: /minerakshak/risk_prediction, /minerakshak/highest_threat, /minerakshak/sensor_health
  ├── FastAPI REST: /api/minerakshak/latest_risk, /api/dashboard
  └── WebSockets: /ws/dashboard, /ws/minerakshak/risk
```

### Sensor Health States vs. ML Risk Tiers

The system strictly decouples telemetry integrity from collision risk:

| Dimension | States / Tiers | Meaning | Action Policy |
| :--- | :--- | :--- | :--- |
| **Sensor Health** | `OK`, `NO_DATA`, `STALE`, `INVALID`, `DEGRADED` | Physical / telemetry data stream integrity | When not `OK`, enters controlled fault state. Never generates fresh predictions from stale data. |
| **ML Risk Level** | `SAFE`, `CAUTION`, `WARNING`, `CRITICAL` | Situational obstacle collision hazard | Computed only from fresh, valid observations. Zero actuation policy strictly maintained. |

---

## 4. Measured Performance & Latency Benchmarks

### Stage 13 Hardware-Ready Software Benchmark (500 Iterations, 361 Beams, 3 Obstacles):
> **Disclaimer**: SYNTHETIC SOFTWARE BENCHMARK — NOT PHYSICAL HARDWARE LATENCY

| Pipeline Stage | Mean Latency | Median (P50) | P95 Latency | P99 Latency | Real-time Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. LaserScan Validation** | 0.113 ms | 0.113 ms | 0.141 ms | 0.155 ms | ✅ Optimized |
| **2. LaserScan Conversion & Offsets** | 0.378 ms | 0.397 ms | 0.446 ms | 0.516 ms | ✅ Optimized |
| **3. Obstacle Extraction & Clustering** | 0.444 ms | 0.464 ms | 0.546 ms | 0.618 ms | ✅ Optimized |
| **4. Multi-Scan Tracking** | 0.015 ms | 0.014 ms | 0.021 ms | 0.029 ms | ✅ Optimized |
| **5. Feature Mapping** | 0.015 ms | 0.014 ms | 0.019 ms | 0.029 ms | ✅ Optimized |
| **6. Frozen HGB Model Inference** | 8.700 ms | 8.381 ms | 11.898 ms | 12.352 ms | ✅ Optimized |
| **TOTAL SOFTWARE CYCLE** | **9.832 ms** | **9.644 ms** | **12.421 ms** | **14.065 ms** | **✅ PASSED** |

* **Real-time 20 Hz Budget**: 50.0 ms
* **Budget Consumption**: **19.7%** of allocated engineering budget
* **Real-time Headroom**: **+35.93 ms** at P99 (80.3% margin)

---

## 5. Verification Commands

From `mineRakshak-ai/`:

```powershell
# 1. Run dedicated Stage 14 test suite (30/30 tests passing)
.\.venv\Scripts\python.exe -m unittest tests/test_stage14_2d_calibration.py -v

# 2. Run complete automated test suite (149/149 tests passing across all suites)
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

# 3. Run Stage 14 calibration, quality, and latency benchmark
.\.venv\Scripts\python.exe src/benchmark_stage14.py

# 4. Run Stage 14 offline validator CLI
.\.venv\Scripts\python.exe -m src.perception.offline_validator_2d

# 5. Verify model artifact SHA-256 hashes
.\.venv\Scripts\python.exe -c "import hashlib, json; [print(f, hashlib.sha256(open(f, 'rb').read()).hexdigest()) for f in ['models/hgb_model.joblib', 'models/preprocessor.joblib', 'models/feature_schema.json', 'models/label_mapping.json', 'models/xgboost_model.json']]"
```

---

## 6. Real Hardware Integration Procedure

When physical 2D LiDAR hardware is delivered to the test bench:
1. Connect physical LiDAR via Ethernet/USB and install vendor ROS 2 driver (e.g., `sick_safetyscanners`, `urg_node`, or `rplidar_ros`).
2. Verify ROS 2 topic `/scan` is publishing `sensor_msgs/msg/LaserScan` at nominal 15–20 Hz.
3. Validate frame ID, angular range (`angle_min`, `angle_max`), and beam resolution.
4. Verify coordinate orientation (+X forward along truck travel heading, +Y lateral left).
5. Apply measured physical mounting offsets ($X, Y, \text{yaw}$) in `LiDAR2DConfig`.
6. Record rosbag2 datasets under varying mining operating conditions (open pit, haul road, dust plume, vehicle approach).
7. Convert or replay recorded scans through `LiDARScanRecorder` and `LiDARScanReplayer`.
8. Run perception clustering and tracking against ground-truth range poles and target vehicles.
9. Evaluate model risk predictions, false positives, and false negatives.
10. Only then determine whether model fine-tuning or retraining is warranted (Decision Gate: Option B).

---
*MineRakshak AI Safety Team — Stage 14 Validation Artifact*
