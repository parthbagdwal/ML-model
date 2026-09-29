# minerakshak_risk ROS 2 Package

## Overview
`minerakshak_risk` is an `ament_python` ROS 2 package providing perception-to-risk inference for haul trucks in open-pit mining environments. It subscribes to obstacle detections emitted by 3D LiDAR point-cloud clustering nodes, maps them into the MineRakshak feature schema, runs the trained `HistGradientBoostingClassifier` model, and publishes real-time hazard classifications and response actions.

> **Disclaimer**: This package is a prototype perception-to-risk integration based on synthetic LiDAR features. It is not a certified mine-safety or vehicle-control system.

---

## Architecture Flow

```text
3D LiDAR Point Cloud (ROS 2 sensor_msgs/PointCloud2)
        │
        ▼
Upstream Euclidean Clustering / 3D Bounding Box Detection
        │
        ▼
Perception Topic: /perception/detected_objects (std_msgs/String JSON)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│             minerakshak_risk_node                      │
│  - Ingests PerceptionFrame                             │
│  - Extracts 12-feature raw observation per obstacle    │
│  - Strict schema validation & physical bounds check    │
│  - Frozen Preprocessor One-Hot Encoding (17 features)   │
│  - Primary Model (HistGradientBoosting) predict_proba  │
│  - Identifies highest-threat obstacle in frame         │
└────────────────────────────────────────────────────────┘
        │
        ├─────────────────────────────┐
        ▼                             ▼
/risk/assessments             /risk/highest_threat
(Full multi-object payload)   (Single highest-priority threat)
        │                             │
        ▼                             ▼
Telemetry / Dashboard         Heads-Up Display / Audio Alert
```

---

## Topics & Interfaces

### 1. Subscription Topics
* **Primary**: `/minerakshak/object_observations`
* **Alias**: `/perception/detected_objects`
* **Type**: `std_msgs/msg/String` (JSON-encoded single observation or `PerceptionFrame`)
* **Payload Fields (per object)**:
  * `object_id`: string
  * `distance_m`: float ($\ge 0.0$)
  * `object_x_m`, `object_y_m`, `object_z_m`: float
  * `object_width_m`, `object_height_m`, `object_length_m`: float ($> 0.0$)
  * `point_count`: int ($\ge 0$)
  * `relative_velocity_mps`: float (negative = closing)
  * `truck_speed_kmph`: float ($\ge 0.0$)
  * `time_to_collision_s`: float (optional, auto-derived if omitted)
  * `object_type`: string (`car`, `truck`, `crane`, `excavator`, `person`, `unknown`)

### 2. Publication Topics
* **Primary Output**: `/minerakshak/risk_prediction`
* **Alias**: `/risk/assessments`
  * **Type**: `std_msgs/msg/String` (JSON-encoded assessment dictionary or `RiskAssessmentFrame`)
  * **Payload**: Includes `risk_level`, `class_id`, `confidence`, `probabilities` (`SAFE`, `CAUTION`, `WARNING`, `CRITICAL`), `model`, `status`, and `recommendation`.
* **Threat Alert Topic**: `/minerakshak/highest_threat` (or `/risk/highest_threat`)
  * **Type**: `std_msgs/msg/String` (JSON summary of single most severe hazard in current frame for fast-path alerting).

---

## Configuration Parameters (`config/risk_node_params.yaml`)

* `input_topic`: string (default: `/minerakshak/object_observations`)
* `output_topic`: string (default: `/minerakshak/risk_prediction`)
* `threat_topic`: string (default: `/minerakshak/highest_threat`)
* `model_path`: string (default: `models/hgb_model.joblib`)
* `preprocessor_path`: string (default: `models/preprocessor.joblib`)
* `feature_schema_path`: string (default: `models/feature_schema.json`)
* `label_mapping_path`: string (default: `models/label_mapping.json`)
* `enable_secondary_voter`: bool (default: `false`)
* `log_latency`: bool (default: `true`)

---

## Build and Run Instructions

### Prerequisites
* ROS 2 Humble Hawksbill, Iron Irwini, or Rolling Ridley
* Python >= 3.10 with `numpy`, `pandas`, `scikit-learn`, `joblib`, `xgboost`

### Compilation with Colcon
```bash
# From workspace root (e.g. ros2_ws/)
colcon build --packages-select minerakshak_risk

# Source the workspace setup script
# On Linux/macOS:
source install/setup.bash

# On Windows (PowerShell):
.\install\setup.ps1
# or on Windows (cmd):
call install\setup.bat
```

### Launching the Node
```bash
# Direct run via risk_node or minerakshak_inference_node
ros2 run minerakshak_risk minerakshak_inference_node

# Or using the launch file
ros2 launch minerakshak_risk risk_node.launch.py
```

### Testing Topics via CLI
```bash
# In Terminal 1: Echo risk predictions
ros2 topic echo /minerakshak/risk_prediction

# In Terminal 2: Publish a mock detected object
ros2 topic pub --once /minerakshak/object_observations std_msgs/msg/String \
  "{data: '{\"distance_m\":5.5,\"object_x_m\":0.8,\"object_y_m\":5.4,\"object_z_m\":0.3,\"object_width_m\":3.4,\"object_height_m\":3.6,\"object_length_m\":7.5,\"point_count\":750,\"relative_velocity_mps\":-9.0,\"truck_speed_kmph\":32.0,\"time_to_collision_s\":0.61,\"object_type\":\"truck\"}'}"
```
