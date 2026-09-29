# STAGE 13 VALIDATION REPORT — Real 2D LiDAR ROS 2 Interface, Scan Replay & Hardware-Ready Integration

**Project**: MineRakshak AI — Mining Haul Truck Collision Risk Prediction Engine  
**Stage**: Stage 13  
**Date**: September 29, 2026  
**Status**: COMPLETED & RIGOROUSLY VALIDATED (Software & Interface Level)  
**Physical Hardware Status**: **NOT VALIDATED — HARDWARE NOT AVAILABLE**  
**Software Verdict**: **PASSED**  
**Primary ML Model**: Frozen `HistGradientBoostingClassifier` (`models/hgb_model.joblib`)  
**Target Hardware**: 2D Planar LiDAR (ROS 2 `sensor_msgs/msg/LaserScan`)  
**Test Suite**: 119 / 119 Tests Passing (28 Dedicated Stage 13 Tests + 91 Regression Tests)  

---

> [!CAUTION]
> ### CRITICAL HARDWARE & SENSOR PROVENANCE DISCLAIMER
> 1. **HARDWARE-READY / REAL-SENSOR-READY, BUT NOT PHYSICALLY VALIDATED**: Stage 13 provides a complete, robust, hardware-ready ROS 2 `LaserScan` input interface, validation engine, and replay mechanism. However, because physical 2D planar LiDAR sensor hardware was not physically connected or deployed in this environment, this system has **NOT been physically validated**.
> 2. **NO PHYSICAL SENSOR MEASUREMENTS FABRICATED**: Zero synthetic data has been claimed or presented as real physical sensor readings. All test datasets and benchmarks are deterministically synthesized or replayed with explicit provenance headers.
> 3. **STRICT SEPARATION OF SYNTHETIC AND REAL DATA**: Synthetic data and real recorded data are partitioned with strict metadata schemas. Datasets lacking source provenance are rejected.
> 4. **STRICTLY FROZEN ML ARTIFACTS**: The machine learning model (`models/hgb_model.joblib`), preprocessor (`models/preprocessor.joblib`), feature schema (`models/feature_schema.json`), and label mappings (`models/label_mapping.json`) remain 100% bitwise identical to their Stage 4/6/8 baselines. Zero retraining or weight alterations occurred.
> 5. **ZERO VEHICLE ACTUATION INVARIANT**: The system is strictly an advisory decision-support and operator-alerting system. It contains **NO actuation interfaces** and commands **NO braking, steering, throttle, or actuator movements**.

---

## 1. Executive Summary

Stage 13 establishes the production-ready ROS 2 sensor interface layer for MineRakshak AI. It prepares the collision risk prediction engine to consume live planar laser scan data from physical 2D LiDAR hardware via the industry-standard ROS 2 message type `sensor_msgs/msg/LaserScan`.

To guarantee robustness under harsh open-pit and underground mining conditions (airborne dust, diesel particulate slurry, occlusions, and intermittent network latency), Stage 13 introduces:
1. A dedicated `ROS2LaserScanAdapter` providing geometric, bounds, and signal quality validation.
2. An explicit `SensorHealthState` machine (`OK`, `NO_DATA`, `STALE`, `INVALID`, `DEGRADED`) strictly decoupled from situational ML collision hazard levels (`SAFE`, `CAUTION`, `WARNING`, `CRITICAL`).
3. An active stale-scan watchdog that detects stale data (latency > 0.5s) and immediately transitions to a controlled fault state without emitting stale predictions or actuating vehicle equipment.
4. A deterministic scan recording and replay subsystem (`LiDARScanRecorder`, `LiDARScanReplayer`) with mandatory provenance verification (`source_type: synthetic`, `source_type: real_ros2`, `source_type: recorded`).
5. A fully integrated ROS 2 subscription node (`MineRakshakLaserScanNode`) and decoupled pipeline processor (`LaserScanToRiskPipeline`) connecting `/scan` to frozen ML inference and downstream FastAPI/WebSocket dashboards.

Across 119 unit and integration tests (28 dedicated Stage 13 tests and 91 regression tests spanning Stages 1–12), the software passed with a 100.0% success rate. The end-to-end software cycle latency measured **9.67 ms** (mean) and **13.41 ms** (P99), utilizing only 19.3% of the 50.0 ms real-time engineering budget.

---

## 2. Stage 12 Baseline

Stage 12 transitioned the perception layer from legacy 3D point-cloud assumptions to a modular 2D planar LiDAR pipeline. The baseline established:
- Purely planar abstractions: `LaserScan2D`, `LaserScanPoint2D`, `Obstacle2D`, `PerceptionSource`.
- Segmented clustering using Euclidean jump-distance thresholding ($D_{\text{thresh}} = 1.2\,\text{m}$, $N_{\min} = 2$).
- Multi-scan temporal tracking (`MultiScanTracker2D`) deriving closing velocity ($\Delta d / \Delta t$) and Time-to-Collision (TTC).
- Feature compatibility layer (`FeatureMapper2D`) mapping 2D observations to the frozen 12-feature schema contract while auditing missing 3D features ($z$, height, object type) as nominal defaults.
- 91 passing regression tests and frozen ML artifact integrity.

---

## 3. Stage 13 Objective

The objectives of Stage 13 are:
1. **Real ROS 2 Input Interface**: Support standard ROS 2 `sensor_msgs/msg/LaserScan` messages directly.
2. **Robust Input Validation**: Reject or isolate malformed geometry, empty arrays, out-of-bounds ranges, NaNs, Infs, and corrupted beam data.
3. **Sensor Health Decoupling**: Introduce explicit sensor health states separated from ML risk tiers.
4. **Stale Data Safety**: Prevent silent reuse of expired sensor readings and enforce safe fault states.
5. **Coordinate Frame & Mounting Offsets**: Enforce vehicle coordinate convention (+X forward, +Y lateral left, Z unmeasured) with configurable mounting yaw rotation and translational offsets.
6. **Scan Replay System**: Enable recording and reproducible replay of scan datasets with explicit source metadata.
7. **Downstream Integration**: Bridge LaserScan processing seamlessly to the frozen HGB model, FastAPI backend, and WebSockets.
8. **Preserve Invariants**: Enforce zero vehicle actuation, zero 3D fabrication, and bitwise ML artifact freezing.

---

## 4. Architecture

```
[Real 2D Planar LiDAR Sensor / rosbag2 Stream]
                     │
                     ▼
             ROS 2 Topic: /scan
        (sensor_msgs/msg/LaserScan)
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │         ROS2LaserScanAdapter (src/perception/)         │
  │  ├── 1. Format Ingestion (ROS2 Msg, Dict, LaserScan2D) │
  │  ├── 2. Structural & Geometric Validation              │
  │  │    - angle_increment > 0, angle_min < angle_max     │
  │  │    - range_min >= 0, range_max > range_min          │
  │  │    - expected vs. received beam count sanity check  │
  │  ├── 3. Signal Bounds & Quality Filter                 │
  │  │    - filter individual NaN, +inf, -inf beams        │
  │  │    - filter out-of-bounds (r < r_min, r > r_max)   │
  │  │    - corrupted beam ratio -> DEGRADED / INVALID     │
  │  ├── 4. Timestamp & Freshness Watchdog                 │
  │  │    - compute age = now - scan_timestamp             │
  │  │    - age > stale_timeout_s (0.5s) -> STALE health   │
  │  ├── 5. Vehicle Coordinate Transformation              │
  │  │    - +X forward, +Y lateral left, Z unmeasured      │
  │  │    - apply mounting yaw rotation and (x, y) offset  │
  │  └── 6. State Machine: OK, NO_DATA, STALE, INVALID,    │
  │         DEGRADED                                       │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼ (Validated LaserScan2D)
  ┌────────────────────────────────────────────────────────┐
  │     ObstacleExtractor2D (Planar Jump-Distance)        │
  │  - Euclidean jump distance threshold D_thresh = 1.2m   │
  │  - Minimum point filter (min_points = 2)               │
  │  - Extract 2D centroid (x, y), span width, radial depth│
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼ (list[Obstacle2D])
  ┌────────────────────────────────────────────────────────┐
  │     MultiScanTracker2D (Temporal Kinematics)           │
  │  - Nearest-neighbor gating (Gate = 4.5m)               │
  │  - Relative closing velocity: delta_d / delta_t        │
  │  - Kinematic Time-to-Collision (TTC)                   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼ (Tracked Obstacles)
  ┌────────────────────────────────────────────────────────┐
  │     FeatureMapper2D (Schema Compatibility Layer)       │
  │  - Maps distance, x, y, width, depth, beam count       │
  │  - Maps tracking velocity, TTC, vehicle odometry speed │
  │  - Assigns nominal defaults: z=0.0m, h=1.5m, type=unk  │
  │  - Transparent audit: "MODEL COMPATIBILITY REQUIRES    │
  │    REAL-SENSOR VALIDATION"                             │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼ (PerceptionFrame - 12 raw features)
  ┌────────────────────────────────────────────────────────┐
  │     Frozen Inference Engine (predict.py)               │
  │  - Preprocessor (models/preprocessor.joblib, 17 feats) │
  │  - HistGradientBoostingClassifier (models/hgb_model)   │
  │  - Class probabilities: SAFE, CAUTION, WARNING, CRIT   │
  │  - Highest-threat selection & recommendation           │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │     Telemetry Dispatch & Integration Layer             │
  │  - ROS 2: /minerakshak/risk_prediction                 │
  │  - ROS 2: /minerakshak/highest_threat                  │
  │  - ROS 2: /minerakshak/sensor_health                   │
  │  - FastAPI: /api/minerakshak/latest_risk, /dashboard   │
  │  - WebSockets: /ws/dashboard, /ws/minerakshak/risk     │
  └────────────────────────────────────────────────────────┘
```

---

## 5. ROS 2 LaserScan Interface

The interface consumes `sensor_msgs/msg/LaserScan` with full specification support:

| Field Name | Type | Unit / Range | Interpretation & Validation |
| :--- | :--- | :--- | :--- |
| `header.stamp` | `builtin_interfaces/Time` | sec, nanosec | Monotonic timestamp. Validated against current clock (`age <= 0.5s`). |
| `header.frame_id` | `string` | UTF-8 | Sensor coordinate frame identifier. Preserved in telemetry. |
| `angle_min` | `float32` | radians ($-\pi$ to $\pi$) | Starting angle of the scan. Must be strictly less than `angle_max`. |
| `angle_max` | `float32` | radians ($-\pi$ to $\pi$) | Ending angle of the scan. Must be strictly greater than `angle_min`. |
| `angle_increment` | `float32` | radians | Angular resolution between beams. Must be strictly positive ($> 0$). |
| `time_increment` | `float32` | seconds | Time between consecutive beam measurements (preserved). |
| `scan_time` | `float32` | seconds | Total cycle time (e.g. 0.05s for 20 Hz). |
| `range_min` | `float32` | meters | Minimum reliable sensing distance (e.g. 0.1m). Must be $\ge 0$. |
| `range_max` | `float32` | meters | Maximum sensing distance (e.g. 80.0m). Must be $> \text{range\_min}$. |
| `ranges[]` | `float32[]` | meters | Array of measured ranges. Validated, filtered for NaNs/Infs. |
| `intensities[]` | `float32[]` | arbitrary units | Echo return reflectivity amplitude. Optional; preserved when present. |

---

## 6. Coordinate Convention

MineRakshak AI enforces a standard haul truck vehicle coordinate convention:
- **Origin**: Center of the truck front bumper at ground level (or sensor mount plane).
- **$+X$**: Longitudinal travel direction (straight ahead along the truck centerline).
- **$+Y$**: Lateral left direction (perpendicular to travel heading).
- **$-Y$**: Lateral right direction.
- **$Z$**: **NOT MEASURED** by 2D planar LiDAR. No elevation or 3D bounding boxes are fabricated.

### Polar-to-Cartesian Transformation with Mounting Offsets:
When physical LiDAR is mounted with a yaw angle offset $\Delta\psi_{\text{mount}}$ and positional displacement $(x_{\text{mount}}, y_{\text{mount}})$:

$$\theta_{\text{veh}} = \theta_{\text{sensor}} + \Delta\psi_{\text{mount}}$$

$$x_{\text{veh}} = r \cos(\theta_{\text{veh}}) + x_{\text{mount}}$$

$$y_{\text{veh}} = r \sin(\theta_{\text{veh}}) + y_{\text{mount}}$$

When offsets are $(0, 0, 0)$, this reduces to pure sensor-centered polar conversion.

---

## 7. Sensor Configuration

Configured via `LiDAR2DConfig` in `src/perception/sensor_config_2d.py`:

```python
@dataclass
class LiDAR2DConfig:
    topic_name: str = "/scan"
    expected_frame_id: str = "laser_frame"
    range_min_m: float = 0.1
    range_max_m: float = 80.0
    expected_fov_deg: float = 180.0
    angular_resolution_deg: float = 0.5
    mounting_yaw_offset_rad: float = 0.0
    mounting_x_offset_m: float = 0.0
    mounting_y_offset_m: float = 0.0
    stale_timeout_s: float = 0.5
    max_timestamp_gap_s: float = 1.0
    min_valid_beam_ratio: float = 0.15
    default_truck_speed_kmph: float = 25.0
```

---

## 8. Sensor Health and Stale Data

Sensor health status is decoupled from ML risk predictions:

| Health State | Condition Trigger | Pipeline Behavior |
| :--- | :--- | :--- |
| `OK` | Fresh scan, valid geometry, $>85\%$ valid beams | Normal obstacle extraction, tracking, and ML inference. |
| `NO_DATA` | System startup; no scans received yet | Idle state; outputs safe neutral baseline; zero actuation. |
| `STALE` | Current time minus scan timestamp $> 0.5\,\text{s}$ | Emits fault frame (`sensor_health="STALE"`), suppresses stale predictions, zero actuation. |
| `INVALID` | Malformed parameters, empty ranges, corrupted array | Emits fault frame (`sensor_health="INVALID"`), isolates fault, zero actuation. |
| `DEGRADED` | $>85\%$ corrupted beams (severe dust plume/occlusion) | Operates with warning flag; filters corrupted beams, zero actuation. |

---

## 9. Scan Replay Architecture

Module: `src/perception/lidar_scan_replay.py`  
Classes: `LiDARScanRecorder`, `LiDARScanReplayer`  
Format: Deterministic JSON Lines (`.jsonl`)

### Provenance Audit Contract:
Line 1 must contain a mandatory metadata header:
```json
{
  "type": "metadata",
  "spec_version": "1.0",
  "source_type": "synthetic",
  "hardware_validated": false,
  "sensor_model": "2D Planar LiDAR",
  "description": "Controlled operational benchmark dataset",
  "created_at_utc": "2026-09-29T05:25:00Z",
  "disclaimer": "HARDWARE-READY / REAL-SENSOR-READY, BUT NOT PHYSICALLY VALIDATED"
}
```

If `source_type` is missing or unrecognized, or if synthetic data is fraudulently flagged as `hardware_validated=True`, the replayer immediately rejects the dataset with an explicit `AmbiguousReplayDatasetError`.

---

## 10. Synthetic vs. Real Data Separation

| Property | Synthetic Test Data | Real Recorded LiDAR Data |
| :--- | :--- | :--- |
| **Origin** | `Synthetic2DLiDARAdapter` | Physical 2D LiDAR driver via ROS 2 `/scan` |
| **Availability** | Available and active | **NOT available in current development environment** |
| **`source_type`** | `"synthetic"` | `"real_ros2"` or `"recorded"` |
| **`hardware_validated`** | `False` | `True` (only after genuine physical test bench runs) |
| **Purpose** | Unit testing, integration validation, benchmarking | Production deployment & field safety sign-off |
| **Safety Rule** | Never mix into hardware validation statistics | Governed by physical mining test protocols |

---

## 11. Frozen ML Artifact Integrity

SHA-256 cryptographic hashes verified before and after implementation:

| Artifact File | Expected Baseline Hash | Post-Implementation Hash | Verification Status |
| :--- | :---: | :---: | :---: |
| `models/hgb_model.joblib` | `f885c29d3ceddae8d2f5d7a2c1ff44fac849ec57ed498504938d632adb7f1974` | `f885c29d3ceddae8d2f5d7a2c1ff44fac849ec57ed498504938d632adb7f1974` | **IDENTICAL (UNMODIFIED)** |
| `models/preprocessor.joblib` | `37bc0e1366174965e741ab223ff3619c2e6fdc62e139b24ba5e8eaff3f0948da` | `37bc0e1366174965e741ab223ff3619c2e6fdc62e139b24ba5e8eaff3f0948da` | **IDENTICAL (UNMODIFIED)** |
| `models/feature_schema.json` | `e9ca72a4b9a9675c8e85ad2059f0448f2140891512ad4d19c10280b90d2f986b` | `e9ca72a4b9a9675c8e85ad2059f0448f2140891512ad4d19c10280b90d2f986b` | **IDENTICAL (UNMODIFIED)** |
| `models/label_mapping.json` | `2b5fd687a4f99cc2fcc94e3bd031c48571615193e7cff4de621a379492b17ebc` | `2b5fd687a4f99cc2fcc94e3bd031c48571615193e7cff4de621a379492b17ebc` | **IDENTICAL (UNMODIFIED)** |
| `models/xgboost_model.json` | `bb385fb728583b385acbd1e7a052a4579a951835473dc10b12f6dccfdfacb983` | `bb385fb728583b385acbd1e7a052a4579a951835473dc10b12f6dccfdfacb983` | **IDENTICAL (UNMODIFIED)** |

Verification metric recorded at: `results/metrics/stage13_artifact_integrity.json`

---

## 12. Test Results

Dedicated Stage 13 Test Suite: `tests/test_stage13_ros2_laserscan.py`  
**28 / 28 Tests Passed (100.0%)**

| Test # | Test Name | Target Behavior | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_01_valid_laserscan_conversion` | Convert ROS 2 LaserScan to LaserScan2D | **PASS** |
| 2 | `test_02_correct_angle_calculation` | Polar beam angle calculation across $-\pi/2$ to $+\pi/2$ | **PASS** |
| 3 | `test_03_correct_range_handling` | Polar-to-Cartesian range conversion accuracy | **PASS** |
| 4 | `test_04_nan_rejection` | Individual NaN return rejection and filtering | **PASS** |
| 5 | `test_05_infinity_rejection` | Safe rejection of $+inf$ and $-inf$ returns | **PASS** |
| 6 | `test_06_out_of_range_rejection` | Out-of-bounds return filtering ($r < 0.1\,\text{m}$, $r > 80.0\,\text{m}$) | **PASS** |
| 7 | `test_07_empty_scan_handling` | Controlled fault handling on empty beam arrays | **PASS** |
| 8 | `test_08_invalid_configuration_handling` | Rejection of bad scan geometry ($\Delta\theta \le 0$, $\theta_{\min} \ge \theta_{\max}$) | **PASS** |
| 9 | `test_09_timestamp_preservation` | Exact preservation of nanosecond timestamps | **PASS** |
| 10 | `test_10_frame_id_preservation` | Preservation of sensor frame ID string | **PASS** |
| 11 | `test_11_mounting_offset_handling` | Mounting yaw rotation and translational offsets | **PASS** |
| 12 | `test_12_stale_scan_detection` | Stale scan detection ($age > 0.5\,\text{s}$) & health transition | **PASS** |
| 13 | `test_13_invalid_scan_detection` | Corrupted array detection and health flagging | **PASS** |
| 14 | `test_14_sensor_health_transitions` | Full health state machine transitions (`NO_DATA` $\to$ `OK` $\to$ `STALE` $\to$ `INVALID` $\to$ `DEGRADED`) | **PASS** |
| 15 | `test_15_replay_serialization` | Deterministic JSON Lines scan recording with metadata | **PASS** |
| 16 | `test_16_replay_deserialization` | Replayer loading and sequential scan streaming | **PASS** |
| 17 | `test_17_synthetic_real_metadata_separation` | Rejection of ambiguous provenance & synthetic mislabeling | **PASS** |
| 18 | `test_18_extractor_compatibility` | Compatibility with `ObstacleExtractor2D` | **PASS** |
| 19 | `test_19_tracker_compatibility` | Compatibility with `MultiScanTracker2D` velocity derivation | **PASS** |
| 20 | `test_20_feature_mapper_compatibility` | Compatibility with `FeatureMapper2D` schema mapping | **PASS** |
| 21 | `test_21_frozen_hgb_inference_compatibility` | Clean execution of frozen HGB inference on 2D inputs | **PASS** |
| 22 | `test_22_highest_threat_selection` | Prioritization of highest threat level in multi-obstacle frames | **PASS** |
| 23 | `test_23_ros2_risk_message_compatibility` | Node callback publishing structured `RiskAssessmentFrame` | **PASS** |
| 24 | `test_24_fastapi_integration_compatibility` | Telemetry forwarding to FastAPI `/api/minerakshak/latest_risk` | **PASS** |
| 25 | `test_25_websocket_integration_compatibility` | Real-time WebSocket delivery of evaluated risk frames | **PASS** |
| 26 | `test_26_zero_actuation_invariant` | Strict absence of braking, steering, or throttle commands | **PASS** |
| 27 | `test_27_frozen_model_hash_verification` | Verification of all 5 frozen artifact SHA-256 hashes | **PASS** |
| 28 | `test_28_no_3d_fabrication_invariant` | Audit metadata confirms 3D features are nominal defaults | **PASS** |

---

## 13. Regression Results

Full Automated Test Suite: `python -m unittest discover -s tests -p "test_*.py" -v`  
**119 / 119 Tests Passed (100.0%)**

- `tests/test_model_serialization.py`: 8 tests passing
- `tests/test_inference.py`: 15 tests passing
- `tests/test_ros2_integration.py`: 18 tests passing
- `tests/test_stage10_safety_boundaries.py`: 15 tests passing
- `tests/test_stage11_end_to_end.py`: 17 tests passing
- `tests/test_stage12_2d_perception.py`: 18 tests passing
- `tests/test_stage13_ros2_laserscan.py`: 28 tests passing
- **Total Regressions**: **0**

---

## 14. Latency Benchmark

Benchmark script: `src/benchmark_stage13.py`  
Sample size: 500 cycles (50 warmup cycles)  
Input: Simulated ROS 2 LaserScan (361 beams, 0.5° resolution, 3 simultaneous obstacles)  
Engineering Latency Budget: **50.0 ms** (20 Hz requirement)

> **Mandatory Benchmark Disclaimer**:  
> SYNTHETIC SOFTWARE BENCHMARK — NOT PHYSICAL HARDWARE LATENCY

| Pipeline Stage | Mean (ms) | Median (P50) | P95 (ms) | P99 (ms) | Budget % | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. LaserScan Validation** | 0.113 | 0.113 | 0.141 | 0.155 | 0.2% | ✅ Optimized |
| **2. LaserScan Conversion & Offsets** | 0.378 | 0.397 | 0.446 | 0.516 | 0.8% | ✅ Optimized |
| **3. Obstacle Extraction & Clustering** | 0.444 | 0.464 | 0.546 | 0.618 | 0.9% | ✅ Optimized |
| **4. Multi-Scan Tracking** | 0.015 | 0.014 | 0.021 | 0.029 | <0.1% | ✅ Optimized |
| **5. Feature Mapping** | 0.015 | 0.014 | 0.019 | 0.029 | <0.1% | ✅ Optimized |
| **6. Frozen HGB Model Inference** | 8.700 | 8.381 | 11.898 | 12.352 | 17.4% | ✅ Optimized |
| **TOTAL SOFTWARE CYCLE** | **9.668** | **9.433** | **13.037** | **13.409** | **19.3%** | **✅ PASSED** |

- **Real-Time Margin**: **+36.59 ms headroom** below the 50 ms budget at the 99th percentile.
- Metrics file: `results/metrics/stage13_benchmark.json`
- Visualization: `results/plots/stage13_latency_breakdown.png`

---

## 15. Known Limitations

1. **Physical Hardware Unavailable**: Physical 2D LiDAR units (e.g. SICK TiM781, Hokuyo UST-20LX, or RPLIDAR) are not currently interfaced. Hardware-level physical validation cannot be claimed.
2. **Missing Native ROS 2 Daemon on Windows SAC**: The Windows environment lacks native ROS 2 packages (`rclpy`, `sensor_msgs`). The codebase provides clean dependency isolation, allowing interface testing and deterministic execution while leaving native runtime deployment to target Linux/ROS 2 environments.
3. **Planar Elevation Inability**: A single 2D planar LiDAR cannot detect vertical dimensions, ground slope, or object height. Nominal contract baselines ($z=0.0\,\text{m}$, $h=1.5\,\text{m}$) remain required to satisfy the frozen inference engine.
4. **Single-Scan Object Classification Unavailable**: 2D contour returns cannot reliably distinguish haul trucks from excavators or cranes without external sensor fusion (e.g. camera/thermal vision).
5. **Vehicle Odometry Dependency**: Truck forward speed (`truck_speed_kmph`) cannot be reliably extracted from a single 2D LiDAR scan and must be supplied by external CAN bus or GPS telemetry.

---

## 16. Real Hardware Integration Procedure

When physical 2D LiDAR hardware becomes available:
1. Mount the 2D LiDAR on the front bumper centerline of the haul truck with unobstructed forward field-of-view.
2. Connect the sensor via ruggedized industrial Ethernet (or USB) to the vehicle onboard computer.
3. Launch the vendor ROS 2 driver (e.g. `ros2 launch sick_scan_xd sick_tim_7xx.launch.py`).
4. Confirm topic `/scan` is actively publishing `sensor_msgs/msg/LaserScan` at nominal 20 Hz using `ros2 topic hz /scan`.
5. Verify sensor mounting parameters in `LiDAR2DConfig` (`mounting_x_offset_m`, `mounting_y_offset_m`, `mounting_yaw_offset_rad`).
6. Record rosbag2 datasets (`ros2 bag record /scan /tf /vehicle_telemetry`) during haul cycles under varying conditions (clear, dust clouds, water spray, vehicle approach).
7. Replay recorded bags through `LiDARScanReplayer` to verify obstacle extraction and tracking accuracy against ground truth.
8. Evaluate model risk classifications and false alarm rates.
9. Only if empirical sensor noise characteristics degrade inference accuracy should model retraining be considered.

---

## 17. Safety Boundaries

All 10 required safety invariants are confirmed:

- **Invariant 1 (Zero Actuator Commands)**: The system emits strictly decision-support alerts. No mechanical actuator commands exist.
- **Invariant 2 (Zero Braking Commands)**: No service brake or retarder commands are generated.
- **Invariant 3 (Zero Steering Commands)**: No steering angle adjustments are commanded.
- **Invariant 4 (Zero Throttle Commands)**: No engine throttle adjustments are commanded.
- **Invariant 5 (No Fabricated Physical Data)**: Synthetic data is strictly declared as synthetic. No fabricated data is represented as real sensor data.
- **Invariant 6 (No Fabricated 3D Measurements)**: Elevation, height, and object type are explicitly audited as nominal contract defaults.
- **Invariant 7 (ML Artifact Inviolability)**: Preprocessor, model weights, schemas, and label mappings are 100% bitwise frozen.
- **Invariant 8 (Stale Sensor Safety)**: Sensor readings older than 0.5s are flagged STALE and never silently converted into fresh predictions.
- **Invariant 9 (Health vs. Risk Decoupling)**: Sensor health state (`STALE`, `INVALID`, `DEGRADED`) is never conflated with ML risk tier (`SAFE`, `CAUTION`, `WARNING`, `CRITICAL`).
- **Invariant 10 (Synthetic Benchmark Integrity)**: Latency benchmarks are explicitly labeled as software-only benchmarks.

---

## 18. Final Verdict

### STAGE 13 SOFTWARE STATUS
**`PASSED`**

### PHYSICAL 2D LIDAR VALIDATION
**`NOT VALIDATED — HARDWARE NOT AVAILABLE`**

---
*Signed: MineRakshak AI Safety & Integration Engineering Team*  
*Stage 13 Validation Sign-Off: September 29, 2026*
