# STAGE 12 VALIDATION REPORT — 2D LiDAR Perception Adapter & Sensor-Ready Integration

**Project**: MineRakshak AI — Mining Haul Truck Collision Risk Prediction Engine  
**Stage**: Stage 12  
**Date**: September 29, 2026  
**Status**: COMPLETED & RIGOROUSLY VALIDATED (Software Integration Level)  
**Primary ML Model**: Frozen `HistGradientBoostingClassifier` (`models/hgb_model.joblib`)  
**Target Hardware**: 2D Planar LiDAR (e.g., SICK / Hokuyo / RPLIDAR / ROS 2 `LaserScan`)  
**Test Suite**: 91 / 91 Tests Passing (18 Dedicated Stage 12 Tests + 73 Regression Tests)  

---

> [!CAUTION]
> ### CRITICAL HARDWARE & OPERATIONAL DISCLAIMER
> 1. **HARDWARE TARGET IS 2D LIDAR**: The actual physical hardware available for this project is a **2D planar LiDAR**, NOT a 3D LiDAR.
> 2. **NO REAL 2D SENSOR READINGS CURRENTLY AVAILABLE**: Physical 2D LiDAR readings have not yet been collected or interfaced in this development stage.
> 3. **SYNTHETIC 2D TEST SOURCE ONLY**: All Stage 12 perception tests and benchmarks utilize a deterministic **Synthetic 2D LiDAR software test harness** (`Synthetic2DLiDARAdapter`). This does **NOT** constitute real-world physical sensor validation.
> 4. **ZERO 3D FABRICATION**: The system strictly refuses to fabricate nonexistent 3D measurements (elevation $z$, height, 3D volumetric bounding boxes, or single-scan object classes) to feed into the model.
> 5. **FROZEN ML MODEL**: The trained model (`hgb_model.joblib`), preprocessor (`preprocessor.joblib`), feature schema (`feature_schema.json`), and label mappings (`label_mapping.json`) remain strictly unmodified and frozen.
> 6. **ZERO VEHICLE ACTUATION**: The system is strictly advisory decision-support and driver-alerting. It commands NO braking, steering, throttle, or actuator movements.

---

## 1. Stage 12 Objective

The core objective of Stage 12 is to transition the perception architecture from legacy 3D point-cloud assumptions to an explicit, modular **2D planar LiDAR perception pipeline**, preparing the system for future physical 2D LiDAR integration without breaking or modifying the frozen Stage 4–11 machine learning inference engine.

Specific milestones achieved:
1. Complete audit of 3D assumptions across the entire codebase.
2. Creation of sensor-independent data structures (`LaserScan2D`, `LaserScanPoint2D`, `Obstacle2D`, `PerceptionSource`).
3. Implementation of a deterministic, planar `Synthetic2DLiDARAdapter` strictly for software testing.
4. Construction of a 2D perception pipeline: beam range filtering, polar-to-Cartesian transformation, neighbor clustering via Euclidean jump-distance thresholding, planar obstacle property calculation, and temporal multi-scan velocity tracking (`MultiScanTracker2D`).
5. Documented feature compatibility analysis comparing 2D LiDAR measurements against the frozen model's 12 raw feature inputs.
6. Implementation of an auditable `FeatureMapper2D` layer that transparently flags missing 3D features while safely satisfying the frozen inference contract.
7. Complete isolation of legacy 3D reference code (`MockLiDARPerceptionAdapter`) for backward regression verification.
8. Implementation of a dedicated 18-test Stage 12 test suite and latency benchmark.

---

## 2. Existing 3D Assumption Audit

An exhaustive audit of the MineRakshak codebase identified every location where 3D assumptions were embedded:

| Component Path | Component Role | 3D Assumption Identified | Categorization | Architectural Disposition |
| :--- | :--- | :--- | :--- | :--- |
| `src/perception/mock_lidar_adapter.py` | Perception Simulation | Simulates 3D point cloud ground-plane removal, 3D bounding boxes ($w, h, l$), elevation $z$, and 3D object classification (`person`, `car`, `truck`). | **Category B (3D-Specific)** | **Isolated as Legacy Reference**. Retained solely for Stage 10/11 regression testing. Renamed documentation to emphasize reference status. |
| `src/data_generation/generate_dataset.py` | Synthetic Data Generation | Synthesized 3D bounding box dimensions $(w, h, l)$, 3D positions $(x, y, z)$, 3D point counts based on surface area/volume. | **Category B (3D-Specific)** | Historical generation script. Kept frozen to preserve reproducibility of Stages 1–3. |
| `src/preprocessing/schema.py` | Feature Contract | Defines `object_z_m`, `object_height_m`, `object_length_m`, and 3D bounding box descriptions. | **Category C (Risk-Model-Specific)** | **Frozen Contract**. Schema cannot be altered without retraining HGB. Serves as target for Stage 12 feature mapper. |
| `models/feature_schema.json` | Model Metadata | Expects 12 raw features including $z$, height, length, 3D point count, and object type. | **Category C (Risk-Model-Specific)** | **Strictly Frozen**. Bitwise identical. |
| `src/inference/predict.py` | Inference Engine | Validates that $w, h, l > 0$, $z$ is numeric, and object type is present. | **Category A & C** | Core inference logic is sensor-independent; input validation checks preserved via Stage 12 mapper. |
| `ros2_ws/schemas.py` | ROS 2 Contracts | `DetectedObject` schema includes fields for $z$, height, and length. | **Category E (ROS 2 Specific)** | Maintained for backward ROS 2 compatibility; 2D mapper populates planar dimensions and nominal contract defaults. |
| `main.py` | Backend Server | Ingests `RiskAssessmentFrame`, updates state, broadcasts over WebSockets. | **Category D (FastAPI Specific)** | **Completely Sensor-Independent**. Operates on risk assessment outputs, unaffected by 2D/3D sensor origin. |

---

## 3. New 2D LiDAR Architecture

Stage 12 establishes a clean, decoupled perception pipeline where the sensor input layer can be swapped from synthetic testing to a real ROS 2 `sensor_msgs/msg/LaserScan` driver without altering risk prediction:

```
[Future Physical 2D LiDAR Hardware]
               │
               ▼ (ROS 2 /scan or Serial/Ethernet)
[LaserScan2D Input Interface]
  ├── angle_min, angle_max, angle_increment
  ├── range_min, range_max
  └── ranges[] array (361 beams @ 0.5° resolution)
               │
               ▼
[2D Obstacle Extraction & Segmentation]
  ├── Invalid Beam Filter (NaN, Inf, Out-of-bounds rejection)
  ├── Polar-to-Cartesian (+X forward, +Y lateral)
  └── Euclidean Jump-Distance Clustering (D_thresh = 1.2m)
               │
               ▼
[Multi-Scan Temporal Tracker (MultiScanTracker2D)]
  ├── Nearest-Neighbor Centroid Association (Gate = 4.5m)
  ├── Delta-Distance / Delta-Time Velocity Derivation (relative_velocity_mps)
  └── Kinematic Time-to-Collision (time_to_collision_s)
               │
               ▼
[Feature Compatibility & Mapping Layer (FeatureMapper2D)]
  ├── Direct 2D Mappings: distance, x, y, span width, depth, beam count
  ├── Kinematic Mappings: relative velocity, TTC, vehicle odometry
  ├── Nominal Contract Defaults: z = 0.0m, height = 1.5m, type = 'unknown'
  └── Transparent Metadata Audit: "MODEL COMPATIBILITY REQUIRES REAL-SENSOR VALIDATION"
               │
               ▼
[Existing Frozen HGB Model Pipeline]
  ├── Preprocessor (models/preprocessor.joblib - 17 features)
  └── HistGradientBoostingClassifier (models/hgb_model.joblib)
               │
               ▼
[Risk Classification & Highest-Threat Selection]
  ├── Four-class probabilities & confidence
  └── Threat hierarchy: CRITICAL > WARNING > CAUTION > SAFE
               │
               ▼
[ROS 2 Node / FastAPI Backend / WebSocket Dashboard]
```

---

## 4. Synthetic 2D LiDAR Adapter

Because real 2D LiDAR readings are not currently available, a deterministic synthetic 2D LiDAR adapter (`Synthetic2DLiDARAdapter`) was constructed in `src/perception/synthetic_lidar_2d.py`.

### Technical Specifications:
* **Field of View**: $180^\circ$ ($-\pi/2$ to $+\pi/2$ radians).
* **Angular Resolution**: $0.5^\circ$ ($0.0087266\text{ rad}$) $\rightarrow 361$ beams per scan.
* **Range Limits**: $r_{\min} = 0.1\text{ m}$, $r_{\max} = 80.0\text{ m}$.
* **Simulated Scan Rate**: $20\text{ Hz}$ ($50\text{ ms}$ interval).
* **Pre-configured Test Scenarios**:
  1. `create_empty_scan()`: Free space across all beams ($r = \infty$ or $r_{\max}$).
  2. `create_single_obstacle_scan()`: Geometric chord projection of a planar obstacle at defined distance, bearing, and width.
  3. `create_safe_scenario_scan()`: Distant pedestrian/object (36m, lateral 5m).
  4. `create_caution_scenario_scan()`: Lateral obstacle in adjacent mining lane (14m, lateral -12m).
  5. `create_warning_scenario_scan()`: Obstacle closing in travel corridor (15m, in-lane).
  6. `create_critical_scenario_scan()`: Imminent head-on hazard (6m, in-lane).
  7. `create_multi_obstacle_scan()`: Multiple simultaneous targets at staggered bearings and depths.
  8. `create_faulty_scan()`: Injection of NaN ranges, infinite ranges, negative ranges, and out-of-boundary measurements.
  9. `create_consecutive_approaching_scans()`: Multi-frame sequence simulating closing dynamics with monotonic timestamps.

---

## 5. 2D Obstacle Extraction

Implemented in `src/perception/obstacle_extractor_2d.py`:

1. **Beam Filtering & Cartesian Conversion**:
   * Rejects any beam with $r < r_{\min}$, $r > r_{\max}$, $\text{isnan}(r)$, or $\text{isinf}(r)$.
   * Projects valid polar beams to vehicle coordinate frame:
     $$x_i = r_i \cos(\theta_i), \quad y_i = r_i \sin(\theta_i)$$
     where $+X$ is forward travel and $+Y$ is lateral left.
2. **Neighbor Clustering via Euclidean Jump-Distance Metric**:
   * Evaluates Euclidean distance between consecutive valid points:
     $$\Delta D = \sqrt{(x_{i} - x_{i-1})^2 + (y_{i} - y_{i-1})^2}$$
   * If $\Delta D \le 1.2\text{ m}$, points belong to the same obstacle.
   * If $\Delta D > 1.2\text{ m}$ or an invalid beam occurs, a cluster boundary is formed.
   * Clusters with $< 2$ points are rejected as sensor noise.
3. **Planar Feature Extraction**:
   * **Centroid**: $(\bar{x}, \bar{y}) = \left(\frac{1}{N}\sum x_i, \frac{1}{N}\sum y_i\right)$
   * **Distance**: $r = \sqrt{\bar{x}^2 + \bar{y}^2}$
   * **Span Width**: Max chord distance $\sqrt{(x_{\text{last}} - x_{\text{first}})^2 + (y_{\text{last}} - y_{\text{first}})^2}$ or lateral extent $\max(y) - \min(y)$
   * **Depth Length**: Radial depth along sensor line-of-sight $\max(x) - \min(x)$
   * **Point Count**: Total number of beam returns $N$
4. **Temporal Multi-Scan Tracking (`MultiScanTracker2D`)**:
   * Associating detections across consecutive scans using nearest-neighbor Euclidean distance gating ($d_{\text{gate}} = 4.5\text{ m}$).
   * Deriving relative closing velocity from distance rate of change:
     $$v_{\text{meas}} = \frac{d_k - d_{k-1}}{\Delta t}$$
   * Exponential smoothing: $v_{\text{smooth}} = 0.7 \cdot v_{\text{meas}} + 0.3 \cdot v_{\text{prev}}$
   * Negative velocity denotes an approaching obstacle (closing in).
   * Derives Time-to-Collision:
     $$\text{TTC} = \begin{cases} \frac{d_k}{-v_{\text{smooth}}}, & \text{if } v_{\text{smooth}} < -0.05\text{ m/s} \\ 99.9\text{ s}, & \text{otherwise} \end{cases}$$

---

## 6. Feature Compatibility Matrix

The following matrix documents the exact compatibility between genuine 2D LiDAR planar measurements and the 12 raw features expected by the frozen ML model:

| Feature Name | 2D LiDAR Availability | Extraction / Derivation Method | Physical Limitation & Engineering Handling |
| :--- | :--- | :--- | :--- |
| `distance_m` | **Available** | Radial Euclidean distance to 2D centroid: $\sqrt{\bar{x}^2 + \bar{y}^2}$ | Measures distance to nearest visible planar cross-section, not 3D volumetric centroid. |
| `object_x_m` | **Available** | Planar longitudinal coordinate $+X$ along haul truck travel heading | Planar forward coordinate only. |
| `object_y_m` | **Available** | Planar lateral coordinate $+/-Y$ relative to haul truck centerline | Planar lateral coordinate only. |
| `object_z_m` | **NOT directly available** | Mapped to $0.0\text{ m}$ nominal sensor mount plane | **Single scan plane cannot measure elevation.** Marked `z_is_real_measurement = False`. |
| `object_width_m` | **Approximate** | Planar cluster spread across beam extremities in scan plane | Measures visible chord width in scan plane; occluded or angled surfaces may underestimate width. |
| `object_height_m` | **NOT available** | Mapped to $1.5\text{ m}$ nominal baseline | **Single scan slice has zero vertical dimension.** Marked `height_is_real_measurement = False`. |
| `object_length_m` | **Approximate / Conditional** | Radial beam depth span along line of sight ($\Delta x$) | Planar beam only detects front surface; trailing vehicle geometry is occluded. |
| `point_count` | **Available** | Number of beam returns in the 2D cluster | Represents 2D beam hit count, not 3D volumetric point cloud count. |
| `relative_velocity_mps` | **Multi-scan derived** | Temporal tracking across scans: $\Delta d / \Delta t$ via `MultiScanTracker2D` | Requires at least 2 consecutive scan cycles. Single scan cannot measure velocity. |
| `truck_speed_kmph` | **External source** | Vehicle CAN bus / GPS / odometry telemetry | External telemetry signal, not measured directly by LiDAR. |
| `time_to_collision_s` | **Derived** | Kinematic formula: $d / (-v)$ when closing, else $99.9\text{ s}$ | Inherits tracking velocity estimation uncertainties. |
| `object_type` | **NOT reliably available** | Mapped to `"unknown"` by default (or external camera tag) | Single 2D contour slice cannot reliably classify vehicles, equipment, or personnel. |

**Exported Machine-Readable Artifact**: `results/metrics/stage12_feature_compatibility.csv`

---

## 7. Frozen Model Verification

The integrity of all machine learning artifacts was audited before and after Stage 12:

| ML Artifact | Original Stage | Current File Path | Status | Verification Checksum / Invariant |
| :--- | :--- | :--- | :--- | :--- |
| Primary Model | Stage 7 | `models/hgb_model.joblib` | **FROZEN (Unmodified)** | Verified bitwise unchanged; loaded and evaluated cleanly |
| Reference Model | Stage 5 | `models/xgboost_model.json` | **FROZEN (Unmodified)** | Verified bitwise unchanged |
| Preprocessor | Stage 4 | `models/preprocessor.joblib` | **FROZEN (Unmodified)** | Frozen fitted transformer producing exactly 17 features |
| Feature Schema | Stage 1 | `models/feature_schema.json` | **FROZEN (Unmodified)** | 12 raw features $\rightarrow$ 17 transformed features |
| Label Mapping | Stage 1 | `models/label_mapping.json` | **FROZEN (Unmodified)** | SAFE:0, CAUTION:1, WARNING:2, CRITICAL:3 |

**Result**: Zero retraining, zero refitting, zero weight modifications, and zero feature schema alterations.

---

## 8. Test Results

The complete test suite was executed across all development stages:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

### Summary of Results:
* **Total Tests Executed**: **91**
* **Total Passed**: **91**
* **Total Failed**: **0**
* **Total Errors**: **0**
* **Success Rate**: **100.0%**
* **Total Execution Time**: **9.494 seconds**

### Test Breakdown by Module:

| Test Suite File | Focus Area | Tests Run | Result |
| :--- | :--- | :---: | :---: |
| `tests/test_stage12_2d_perception.py` | 2D LiDAR conversion, clustering, tracking, mapping, isolation | 18 | **18 / 18 PASS** |
| `tests/test_stage11_end_to_end.py` | Full chain integration (Perception $\rightarrow$ ML $\rightarrow$ FastAPI $\rightarrow$ WebSocket) | 17 | **17 / 17 PASS** |
| `tests/test_stage10_safety_boundaries.py` | Operational scenarios (A–H), safety invariants, fault isolation | 15 | **15 / 15 PASS** |
| `tests/test_ros2_integration.py` | ROS 2 topics, serialization, single & batch object handling | 18 | **18 / 18 PASS** |
| `tests/test_inference.py` | Standalone inference engine, validation bounds, probability sanity | 15 | **15 / 15 PASS** |
| `tests/test_model_serialization.py` | Artifact persistence, schema ordering, reproducibility | 8 | **8 / 8 PASS** |

---

## 9. Benchmark Results

A dedicated 500-iteration benchmark was executed on the synthetic 2D perception pipeline:

```powershell
.\.venv\Scripts\python.exe src/benchmark_stage12.py
```

### Measured Component Latencies (500 Iterations):

| Pipeline Stage | Mean Latency | Median (P50) | P90 Latency | P95 Latency | P99 Latency | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Point Conversion & Filtering** | 0.405 ms | 0.417 ms | 0.455 ms | 0.480 ms | 0.533 ms | 0.812 ms |
| **2D Obstacle Clustering** | 0.456 ms | 0.467 ms | 0.518 ms | 0.555 ms | 0.612 ms | 0.945 ms |
| **Multi-Scan Tracking** | 0.016 ms | 0.015 ms | 0.019 ms | 0.022 ms | 0.029 ms | 0.088 ms |
| **Feature Compatibility Mapping** | 0.015 ms | 0.014 ms | 0.017 ms | 0.020 ms | 0.028 ms | 0.071 ms |
| **Frozen HGB Model Inference** | 9.030 ms | 8.791 ms | 10.635 ms | 11.495 ms | 12.965 ms | 16.210 ms |
| **End-to-End Perception-to-Risk** | **9.922 ms** | **9.710 ms** | **11.644 ms** | **12.447 ms** | **13.985 ms** | **18.126 ms** |

### Latency Budget Assessment:
* **Real-Time 20 Hz Cycle Budget**: $50.0\text{ ms}$
* **Measured End-to-End Mean**: **$9.92\text{ ms}$** ($19.8\%$ budget utilization)
* **Measured P95 Latency**: **$12.45\text{ ms}$** ($24.9\%$ budget utilization)
* **Real-Time Headroom**: **$+37.55\text{ ms}$ margin**
* **Generated Latency Plot**: `results/plots/stage12_2d_latency.png`
* **Generated Metrics Artifact**: `results/metrics/stage12_2d_benchmark.json`

---

## 10. Remaining Limitations

1. **Synthetic Data Boundary**: The 2D LiDAR scans processed in Stage 12 were generated by a mathematical simulator. Real optical absorption from mining dust, airborne slurry, vehicle vibrations, and ground pitch variations were not present in this dataset.
2. **Missing Vertical Dimension**: A planar 2D LiDAR beam cannot observe object height ($z$). A boulder resting flat on the ground and an excavator boom hanging overhead could project similar planar profiles if intersected at the sensor height.
3. **Single-Scan Classification Infeasibility**: A 2D contour slice cannot distinguish whether an obstacle is a mining pickup, haul truck, or personnel without temporal shape tracking, multi-beam lidar, or camera-fusion assistance.
4. **Occlusion & Grazing Angles**: 2D beams only illuminate the facing side of an obstacle. Bounding box length along the radial beam axis reflects depth of visibility, not necessarily physical object length.

---

## 11. Real 2D LiDAR Integration Plan

When physical 2D LiDAR hardware is delivered, integration proceeds via the clean boundary established in Stage 12:

1. **Hardware Driver Launch**:
   * Deploy ROS 2 driver (e.g. `urg_node`, `rplidar_ros`, or SICK driver) publishing to standard `/scan` (`sensor_msgs/msg/LaserScan`).
2. **ROS 2 Adapter Ingestion**:
   * The ROS 2 subscriber ingests `/scan`, directly constructing a `LaserScan2D` instance via `LaserScan2D.from_ros2_laser_scan_dict()`.
3. **Real-Time Clustering & Tracking**:
   * Feed `LaserScan2D` into `ObstacleExtractor2D.extract_obstacles()` and `MultiScanTracker2D.track()`.
4. **Calibration & Gating Tuning**:
   * Empirically calibrate `cluster_distance_threshold_m` against mine dust backscatter and road roughness.
5. **Sensor-Model Compatibility Evaluation**:
   * Collect real mine 2D LiDAR scan recordings (`rosbag2`).
   * Evaluate whether 2D-derived features yield acceptable risk prediction distribution with the frozen HGB model.

---

## 12. Whether Model Retraining Is Currently Justified

### Current Finding:
**MODEL RETRAINING IS NOT CURRENTLY JUSTIFIED IN STAGE 12.**

### Rationale:
1. **Lack of Real Sensor Ground Truth**: Retraining the model now would require training on *synthetic 2D data*. Retraining a synthetic model on a different synthetic distribution without real physical sensor telemetry provides no genuine safety gain and risks discarding the validated Stage 7–11 performance benchmarks.
2. **Frozen Model Operates Safely on 2D Mapped Features**: The compatibility mapping layer successfully drives the frozen model: distance, forward $x$, lateral $y$, relative closing velocity, truck speed, and TTC remain the dominant decision drivers in the gradient-boosted trees.
3. **Formal Stop Condition**: Retraining or redesigning the feature schema must be considered **only after real physical 2D LiDAR telemetry has been gathered** and empirical classification discrepancies are recorded.

---

## 13. Final Stage 12 Verdict

```
===============================================================================
                     STAGE 12 FINAL VERDICT
===============================================================================

  SOFTWARE INTEGRATION:              VALIDATED (PASSED 91/91 TESTS)
  2D PERCEPTION PIPELINE:            VALIDATED (SUB-10ms LATENCY)
  FROZEN ML MODEL INTEGRITY:         CONFIRMED UNMODIFIED
  ZERO-ACTUATION SAFETY INVARIANT:   CONFIRMED ENFORCED

  REAL 2D LIDAR HARDWARE:            NOT VALIDATED
                                     (Awaiting Physical Sensor Hardware)

===============================================================================
```

**Conclusion**: The MineRakshak AI perception layer has successfully transitioned to an explicit, modular 2D planar LiDAR architecture. The software integration is verified, fully functional, and ready to accept live ROS 2 `LaserScan` data once hardware becomes available.
