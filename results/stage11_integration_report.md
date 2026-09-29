# Stage 11: Full System Integration Report

**Document ID:** MR-VAL-STG11-001  
**Project:** MineRakshak AI — Mining Haul Truck Safety Risk Prediction Engine  
**Stage:** 11 — Full System Integration (Perception → ROS 2 → ML → FastAPI → WebSocket)  
**Primary Production Model:** `HistGradientBoostingClassifier` (`models/hgb_model.joblib`)  
**Secondary Reference Model:** `XGBClassifier` (`models/xgboost_model.json`)  
**Frozen Preprocessor:** `MineRakshakPreprocessor` (`models/preprocessor.joblib`, 17 transformed features)  
**Execution Timestamp:** 2026-09-28T19:48:15Z  
**Verdict:** **STAGE 11 FULL SYSTEM INTEGRATION PASSED (73/73 Tests Passing, 8/8 Scenarios Verified, Real-Time Budget Compliant)**

---

> [!CAUTION]
> ### Crucial Operational Safety Disclaimer
> 1. **Synthetic Data Notice**: All models, pipelines, and benchmarks were developed and evaluated strictly using **synthetic sensor features** as an engineering proxy. The system has **NOT** been certified or physically field-tested on haul trucks in active mining operations.
> 2. **Decision Support & Driver Alerting Only**: The system is strictly an **advisory decision-support and driver-alerting platform**. It **CANNOT and DOES NOT** actuate vehicle brakes, steer the vehicle, alter throttle, or control mechanical equipment. No emergency braking or steering commands are generated, transmitted, or interfaced with vehicle actuators.

---

## 1. Executive Summary

Stage 11 accomplishes the full-system software integration of the MineRakshak safety stack, connecting the upstream perception layer to the operator-facing dashboard and REST API:

```
[3D LiDAR / Perception Adapter]
           │
           ▼
[ROS 2 Topic: /minerakshak/object_observations]
           │
           ▼
[Frozen Preprocessor (17 Features) & HistGradientBoosting Model]
           │
           ▼
[ROS 2 Topics: /minerakshak/risk_prediction & /minerakshak/highest_threat]
           │
           ▼
[FastAPI Backend: /api/minerakshak/* & /api/dashboard]
           │
           ▼
[WebSockets: /ws/dashboard & /ws/minerakshak/risk]
```

### Key Milestone Achievements
1. **Unmodified ML Core**: The primary model weights (`hgb_model.joblib`), secondary reference model (`xgboost_model.json`), and preprocessor (`preprocessor.joblib`) were maintained 100% frozen with zero retraining or refitting.
2. **Isolated Perception Adapter**: Developed `MockLiDARPerceptionAdapter` simulating upstream ground-plane removal and Euclidean clustering while explicitly labeled as a synthetic proxy.
3. **Dual ROS 2 Bridges**: Constructed bridges for perception input dispatch and risk output forwarding to the FastAPI telemetry service.
4. **Backend & WebSocket Enhancement**: Augmented the FastAPI backend (`main.py`) with 4 dedicated REST endpoints and a high-frequency WebSocket channel (`/ws/minerakshak/risk`), while embedding real-time ML risk states into the existing `/api/dashboard` and `/ws/dashboard` streams.
5. **Real-Time Conformance Across 5 Stages**: Evaluated full software path latency from perception generation to WebSocket delivery across 1, 5, 10, 20, 50, and 100 obstacles. At 100 obstacles, the complete end-to-end P95 latency was **20.24 ms** (consuming **40.5%** of the $50.0\text{ ms}$ real-time cycle budget).
6. **100% Automated Test Suite Conformance**: All 73 tests (41 from Stages 1–9, 15 from Stage 10, and 17 new Stage 11 integration tests) passed with zero errors or failures.

---

## 2. Workspace & Architecture Audits

### 2.1 Backend Audit (`c:\Users\ADMIN\Downloads\mineRakshak_backend`)
Prior to Stage 11 changes, the backend consisted of a FastAPI service in `main.py` with:
- **Application Entry Point**: `main:app` with CORS middleware.
- **Storage Layer**: SQLite (`minerakshak.db`) managing `trips`, `telemetry`, and `current_state`.
- **Existing Telemetry Logic**: Basic heuristic checks based on visibility distance, speed, and heavy vehicle proximity.
- **Missing Elements**: The backend had zero knowledge of MineRakshak ML risk models, 17-feature schema, or ROS 2 risk topics.
- **Integration Action**: Added REST endpoints (`POST /api/minerakshak/risk_prediction`, `POST /api/minerakshak/highest_threat`, `GET /api/minerakshak/latest_risk`, `GET /api/minerakshak/health`) and dedicated WebSocket (`/ws/minerakshak/risk`). Augmented `get_current_state()` with `minerakshak_risk` payload so existing dashboards automatically receive ML risk without schema breaks.

### 2.2 Perception Layer Audit
- **Current State**: Real-time 3D LiDAR Ethernet/UDP streaming, ground plane removal (RANSAC/Patchwork), and Euclidean clustering were confirmed not to be physically present on this host.
- **Integration Action**: In strict accordance with engineering constraints, created `MockLiDARPerceptionAdapter` (`src/perception/mock_lidar_adapter.py`) as an isolated simulation adapter. It formats obstacle clusters into valid `DetectedObject` and `PerceptionFrame` schemas conforming to the Stage 9/10 contracts.

---

## 3. End-to-End Data Flow & Interface Specifications

### 3.1 Topic & API Mappings

| Stage | Interface / Topic | Protocol | Payload Type | Description |
| :--- | :--- | :---: | :--- | :--- |
| **Perception** | `MockLiDARPerceptionAdapter` | Python Class | `PerceptionFrame` | Simulates 3D bounding boxes, range, velocities |
| **ROS 2 Ingest** | `/minerakshak/object_observations` | `std_msgs/String` | JSON (`PerceptionFrame`) | Subscribed by `MineRakshakRiskNode` |
| **Inference** | `PerceptionToRiskProcessor` | In-Process ML | Feature Vector (17) | Transforms 12 raw fields, infers 4-class probabilities |
| **ROS 2 Output** | `/minerakshak/risk_prediction` | `std_msgs/String` | JSON (`RiskAssessmentFrame`) | Detailed individual object risk evaluations |
| **ROS 2 Output** | `/minerakshak/highest_threat` | `std_msgs/String` | JSON (Threat Alert) | Cycle top hazard alert & recommended action |
| **FastAPI Ingest**| `POST /api/minerakshak/risk_prediction` | HTTP / In-Process | JSON | Ingests risk assessments into backend memory & DB |
| **FastAPI Ingest**| `POST /api/minerakshak/highest_threat` | HTTP / In-Process | JSON | Updates active threat alert in backend |
| **REST Expose** | `GET /api/minerakshak/latest_risk` | HTTP GET | JSON | Exposes full latest risk frame & individual evaluations |
| **REST Expose** | `GET /api/dashboard` | HTTP GET | JSON | Integrated vehicle telemetry + MineRakshak risk state |
| **WebSocket** | `/ws/dashboard` | WebSocket | JSON (`dashboard_update`) | Real-time push to main fleet dashboard |
| **WebSocket** | `/ws/minerakshak/risk` | WebSocket | JSON (`risk_assessment_frame`)| High-frequency real-time ML risk feed |

---

## 4. End-to-End Scenario Verification (Scenarios A through H)

The complete pipeline was evaluated across 8 operational scenarios:

| Scenario | Name / Description | Expected Risk | Predicted Risk | E2E Latency | FastAPI Status | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Scenario A** | SAFE: Distant Pedestrian (36.2m, receding) | `SAFE` | `SAFE` | 30.52 ms | in_process_dispatched | **PASS** |
| **Scenario B** | CAUTION: Lateral Vehicle (12.8m, lateral zone) | `CAUTION` | `CAUTION` | 10.73 ms | in_process_dispatched | **PASS** |
| **Scenario C** | WARNING: Worker in Haul Lane (15.7m, closing) | `WARNING` | `WARNING` | 12.80 ms | in_process_dispatched | **PASS** |
| **Scenario D** | CRITICAL: Head-on Imminent Car (6.3m, head-on) | `CRITICAL` | `CRITICAL` | 9.34 ms | in_process_dispatched | **PASS** |
| **Scenario E** | Mixed Quad Frame (SAFE, CAUTION, WARNING, CRITICAL) | `CRITICAL` | `CRITICAL` | 10.04 ms | in_process_dispatched | **PASS** |
| **Scenario F** | Invalid Sensor Input (-15m negative distance) | `INVALID` | `INVALID` | 0.41 ms | in_process_dispatched | **PASS** |
| **Scenario G** | Unknown Type (`autonomous_inspection_drone`) | `SAFE` | `SAFE` | 9.76 ms | in_process_dispatched | **PASS** |
| **Scenario H** | Simultaneous Threats (WARNING + CRITICAL in frame) | `CRITICAL` | `CRITICAL` | 9.57 ms | in_process_dispatched | **PASS** |

---

## 5. End-to-End Latency Benchmark & Perception Budget

The complete 5-stage software pipeline was profiled across frames containing 1, 5, 10, 20, 50, and 100 detected obstacles over 150 benchmark cycles per count after 25 warmup iterations.

### Latency Percentiles & Stage Breakdown Table

| Objects | Mean E2E (ms) | P50 E2E (ms) | P90 E2E (ms) | P95 E2E (ms) | P99 E2E (ms) | Max E2E (ms) | ML Inference (ms) | FastAPI / WS (ms) | Budget % | Headroom | Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 9.39 | 9.04 | 11.23 | **12.00** | 13.58 | 14.39 | 9.31 | 0.02 | 24.0% | +38.00 ms | 106.5 obj/s |
| **5** | 9.41 | 9.06 | 11.45 | **12.60** | 13.01 | 13.25 | 9.30 | 0.02 | 25.2% | +37.40 ms | 531.3 obj/s |
| **10** | 9.82 | 9.63 | 11.89 | **12.33** | 13.41 | 13.78 | 9.66 | 0.02 | 24.7% | +37.67 ms | 1,018.3 obj/s |
| **20** | 10.41 | 10.31 | 11.95 | **12.54** | 13.78 | 13.92 | 10.16 | 0.02 | 25.1% | +37.46 ms | 1,921.2 obj/s |
| **50** | 12.99 | 12.63 | 15.11 | **16.29** | 21.32 | 22.45 | 12.47 | 0.03 | 32.6% | +33.71 ms | 3,849.1 obj/s |
| **100** | 16.19 | 16.10 | 18.92 | **20.24** | 23.77 | 27.91 | 15.23 | 0.04 | **40.5%** | **+29.76 ms** | **6,176.6 obj/s** |

### Complete Latency Scaling Visualization

![MineRakshak AI Stage 11 Latency Scaling Plot](file:///c:/Users/ADMIN/Downloads/mineRakshak_backend/mineRakshak-ai/results/plots/stage11_e2e_latency.png)

### Key Performance Findings
1. **Full-Pipeline Budget Conformance**: The complete software path (Perception → ROS 2 → ML → FastAPI → WebSocket) consumes only **20.24 ms at P95** for 100 obstacles, satisfying the $50\text{ ms}$ real-time budget with nearly **30 ms of headroom**.
2. **Minimal Bridge Overhead**: The ROS 2 → FastAPI dispatch and WebSocket broadcast introduce less than $0.05\text{ ms}$ of overhead, ensuring zero latency penalty for dashboard integration.
3. **ML Dominance**: The vectorized HGB forward pass accounts for $\sim 94\%$ of the runtime ($15.23\text{ ms}$ at 100 objects), confirming that software framing and middleware add negligible delay.

---

## 6. Reliability and Fault-Isolation Audit

The system was evaluated against 8 fault modes in `tests/test_stage11_end_to_end.py`:

| Fault Condition | Expected System Behavior | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| **Malformed Perception Message** | Corrupt JSON trapped; returns `status="error"`, node remains alive. | Error logged, node stayed active, returned `invalid_frame`. | **PASS** |
| **Invalid Sensor Readings** | Negative range/dimensions rejected before inference. | Object isolated (`status="error"`), other objects evaluated. | **PASS** |
| **NaN / Inf Kinematics** | Rejected by physical validator; error recorded. | Trapped with error status, never evaluated by model. | **PASS** |
| **Backend Unreachable** | Remote FastAPI HTTP failure trapped gracefully. | ML evaluation succeeded; bridge reported transport error without crash. | **PASS** |
| **WebSocket Disconnect** | Abrupt client disconnect handled cleanly. | Dead connection removed by `ConnectionManager`; no exception. | **PASS** |
| **Stale / Duplicate Timestamps**| Processed without error; monotonicity preserved. | Evaluated successfully with correct threat ratings. | **PASS** |
| **Empty Perception Frame** | Evaluates 0 objects without index errors. | Handled safely, highest threat defaulted to `SAFE`. | **PASS** |
| **Novel Object Category** | Defaults to all-zero one-hot representation. | Valid classification produced; probabilities sum to 1.0. | **PASS** |

---

## 7. Safety Boundary Verification

| Safety Boundary Rule | Verification Method | Result | Status |
| :--- | :--- | :--- | :---: |
| **Rule 1: CRITICAL Never Downgraded** | Permutations of CRITICAL alongside SAFE/CAUTION/WARNING obstacles. | Highest threat remained `CRITICAL` in 100% of trials. | **PASS** |
| **Rule 2: WARNING Never Downgraded to SAFE** | Combinations of WARNING and multiple SAFE obstacles. | Highest threat remained `WARNING` in 100% of trials. | **PASS** |
| **Rule 3: Invalid Sensor Data Fault Isolation** | Negative distance, negative point count, NaN speed, infinite distance. | All returned `status="error"` and `class_id=-1`. None returned `valid`. | **PASS** |
| **Rule 4: Zero Vehicle Actuation Commands** | Scanned all output dictionaries, schemas, and endpoints for 14 actuation keys. | Zero actuation keys detected across any layer. | **PASS** |

---

## 8. Automated Test Suite Execution Results

All test suites were executed using Python's standard `unittest` discovery:

```
Command: & ".\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

### Cumulative Test Count Breakdown

| Test Suite File | Stage | Focus Area | Tests Run | Passed | Failed |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `tests/test_inference.py` | 8 | Production Inference Engine | 12 | 12 | 0 |
| `tests/test_model_serialization.py` | 8 | Artifact Persistence & Schemas | 11 | 11 | 0 |
| `tests/test_ros2_integration.py` | 9 | ROS 2 Perception Node & Frames | 18 | 18 | 0 |
| `tests/test_stage10_safety_boundaries.py` | 10 | System Scenarios & Safety Invariants | 15 | 15 | 0 |
| `tests/test_stage11_end_to_end.py` | 11 | Full-System Perception to Dashboard | 17 | 17 | 0 |
| **Total Automated Tests** | **Stages 1–11** | **Complete System Test Suite** | **73** | **73** | **0** |

**Execution Time:** 9.30 seconds  
**Pass Rate:** **100.0%** (73/73 passing)

---

## 9. Operational Guidelines & Commands

### Run End-to-End System Demonstration
```powershell
& ".\.venv\Scripts\python.exe" -u "src/benchmark_stage11.py"
```

### Run Complete Automated Test Suite (73 Tests)
```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

### Launch FastAPI Backend
From workspace root:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Endpoints:
- API Swagger Docs: `http://127.0.0.1:8000/docs`
- Latest Risk Telemetry: `http://127.0.0.1:8000/api/minerakshak/latest_risk`
- Health Check: `http://127.0.0.1:8000/api/minerakshak/health`
- WebSocket Feed: `ws://127.0.0.1:8000/ws/minerakshak/risk`

---

## 10. Final Stage 11 Verdict & Next Steps

### **VERDICT: STAGE 11 FULL SYSTEM INTEGRATION PASSED**

1. **Complete Software Path**: Full perception-to-telemetry data path verified from simulated obstacle clusters through ROS 2, ML inference, FastAPI, and WebSocket push.
2. **Zero Invariant Violations**: Safety boundaries and zero-actuation policies strictly enforced.
3. **Budget Compliant**: Total E2E P95 latency is $20.24\text{ ms}$ at 100 objects, leaving nearly $30\text{ ms}$ of headroom.
4. **Zero Regressions**: 73/73 automated tests passing.

---
*Report certified by MineRakshak AI System Validation Team on 2026-09-28.*
