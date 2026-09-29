"""Stage 9: Perception-to-Risk Integration Demo Script.

Demonstrates the full ROS 2 perception pipeline processing 6 representative obstacles:
  A. SAFE-like object (distant stationary light vehicle)
  B. CAUTION-like object (working excavator at moderate distance)
  C. WARNING-like object (mobile crane on closing trajectory)
  D. CRITICAL-like object (head-on haul truck in imminent collision corridor)
  E. Unknown object type (novel unmodeled category with graceful fallback)
  F. Invalid observation (malformed sensor glitch safely isolated)

Captures outputs and saves to results/metrics/ros2_demo_outputs.json.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROS2_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PKG_ROOT))

from minerakshak_risk.risk_node import (
    MineRakshakRiskNode,
    PerceptionToRiskProcessor,
    ROS2_AVAILABLE,
)
from minerakshak_risk.schemas import (
    DetectedObject,
    PerceptionFrame,
    RiskAssessmentFrame,
)
from src.inference.predict import SAFETY_DISCLAIMER

OUTPUT_DEMO_JSON = PROJECT_ROOT / "results" / "metrics" / "ros2_demo_outputs.json"


def run_pipeline_demo() -> dict:
    """Execute end-to-end perception-to-risk demo pipeline."""
    print("=" * 80)
    print("mineRakshak-ai: Stage 9 ROS 2 Perception-to-Risk Integration Demo")
    print("=" * 80)

    # 1. Initialize Processor and Node
    processor = PerceptionToRiskProcessor()
    node = MineRakshakRiskNode()
    print(f"[*] Node initialized: '{node.node_name}' (ROS 2 Available on host: {ROS2_AVAILABLE})")
    print(f"[*] Primary ML Engine: {processor.engine.primary_model.__class__.__name__}")
    print(f"[*] Frozen Feature Count: {len(processor.engine.feature_schema['transformed_feature_names'])}")
    print("-" * 80)

    # 2. Define the 6 Representative Scenarios
    scenarios = {
        "A_SAFE": DetectedObject(
            object_id="obs_A_safe_car",
            distance_m=75.0,
            object_x_m=8.0,
            object_y_m=74.5,
            object_z_m=0.0,
            object_width_m=2.0,
            object_height_m=1.6,
            object_length_m=4.5,
            point_count=85,
            relative_velocity_mps=0.0,
            truck_speed_kmph=18.0,
            time_to_collision_s=99.9,
            object_type="car",
        ),
        "B_CAUTION": DetectedObject(
            object_id="obs_B_caution_excavator",
            distance_m=35.0,
            object_x_m=6.5,
            object_y_m=34.3,
            object_z_m=1.0,
            object_width_m=4.2,
            object_height_m=3.8,
            object_length_m=6.5,
            point_count=450,
            relative_velocity_mps=-1.2,
            truck_speed_kmph=22.0,
            time_to_collision_s=29.17,
            object_type="excavator",
        ),
        "C_WARNING": DetectedObject(
            object_id="obs_C_warning_crane",
            distance_m=24.0,
            object_x_m=-3.5,
            object_y_m=23.7,
            object_z_m=0.5,
            object_width_m=3.2,
            object_height_m=3.8,
            object_length_m=8.0,
            point_count=520,
            relative_velocity_mps=-4.8,
            truck_speed_kmph=26.0,
            time_to_collision_s=5.0,
            object_type="crane",
        ),
        "D_CRITICAL": DetectedObject(
            object_id="obs_D_critical_truck",
            distance_m=6.5,
            object_x_m=1.2,
            object_y_m=6.3,
            object_z_m=0.3,
            object_width_m=3.5,
            object_height_m=3.8,
            object_length_m=7.5,
            point_count=780,
            relative_velocity_mps=-9.5,
            truck_speed_kmph=32.0,
            time_to_collision_s=0.68,
            object_type="truck",
        ),
        "E_UNKNOWN_TYPE": DetectedObject(
            object_id="obs_E_novel_drone",
            distance_m=22.0,
            object_x_m=0.5,
            object_y_m=21.9,
            object_z_m=2.0,
            object_width_m=1.8,
            object_height_m=1.2,
            object_length_m=1.8,
            point_count=210,
            relative_velocity_mps=-2.5,
            truck_speed_kmph=20.0,
            time_to_collision_s=8.8,
            object_type="unregistered_autonomous_drone",
        ),
        "F_INVALID_OBSERVATION": DetectedObject(
            object_id="obs_F_malformed_sensor_glitch",
            distance_m=-15.0,  # Physically impossible negative range
            object_x_m=0.0,
            object_y_m=0.0,
            object_z_m=0.0,
            object_width_m=-2.0,  # Negative dimension
            object_height_m=1.5,
            object_length_m=2.0,
            point_count=50,
            relative_velocity_mps=-5.0,
            truck_speed_kmph=25.0,
            time_to_collision_s=-1.0,
            object_type="truck",
        ),
    }

    # 3. Assemble Unified Multi-Object Perception Frame
    all_objects = list(scenarios.values())
    frame = PerceptionFrame(
        frame_id="demo_perception_frame_001",
        timestamp_ns=int(time.time() * 1e9),
        truck_speed_kmph=28.0,
        objects=all_objects,
    )

    # 4. Execute Pipeline Frame Processing
    t_start = time.perf_counter()
    res: RiskAssessmentFrame = processor.process_frame(frame)
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    # 5. Display Formatted Results
    print(f"\n[FRAME EVALUATION SUMMARY]")
    print(f"  Frame ID:                 {res.frame_id}")
    print(f"  Objects Ingested:         {res.objects_detected}")
    print(f"  Objects Evaluated:        {res.objects_evaluated}")
    print(f"  Faults Trapped & Logged:  {len(res.errors)}")
    print(f"  Highest Threat Obstacle:  {res.highest_threat_object_id} ({res.highest_threat_level})")
    print(f"  Total Cycle Time:         {total_time_ms:.3f} ms")
    print("=" * 80)

    print(f"{'Scenario':<16} | {'Object ID':<26} | {'Risk Tier':<9} | {'Code':<4} | {'Conf':<6} | {'Latency':<8} | {'Recommended Action'}")
    print("-" * 115)

    captured_assessments = {}
    for a in res.assessments:
        print(f"{a.object_id.split('_')[1]:<16} | {a.object_id:<26} | {a.predicted_risk_level:<9} | {a.risk_code:<4} | {a.confidence:.4f} | {a.inference_latency_ms:.2f} ms | {a.recommended_action}")
        captured_assessments[a.object_id] = {
            "object_id": a.object_id,
            "predicted_risk_level": a.predicted_risk_level,
            "risk_code": a.risk_code,
            "confidence": round(a.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in a.probabilities.items()},
            "recommended_action": a.recommended_action,
            "inference_latency_ms": round(a.inference_latency_ms, 3),
        }

    print("-" * 115)
    print(f"[FAULT ISOLATION AUDIT]")
    captured_errors = []
    for err in res.errors:
        print(f"  [TRAPPED] Object '{err['object_id']}': {err['error']}")
        captured_errors.append(err)

    # 6. Build Demo Artifact JSON
    demo_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ros2_available_on_host": ROS2_AVAILABLE,
        "demonstration_scenarios": {
            "scenario_A": {"description": "Distant light vehicle (SAFE)", "input": scenarios["A_SAFE"].__dict__},
            "scenario_B": {"description": "Working excavator at distance (CAUTION)", "input": scenarios["B_CAUTION"].__dict__},
            "scenario_C": {"description": "Approaching crane (WARNING)", "input": scenarios["C_WARNING"].__dict__},
            "scenario_D": {"description": "Head-on collision trajectory (CRITICAL)", "input": scenarios["D_CRITICAL"].__dict__},
            "scenario_E": {"description": "Novel drone category (UNKNOWN TYPE)", "input": scenarios["E_UNKNOWN_TYPE"].__dict__},
            "scenario_F": {"description": "Corrupted sensor reading (INVALID)", "input": scenarios["F_INVALID_OBSERVATION"].__dict__},
        },
        "frame_results": {
            "frame_id": res.frame_id,
            "objects_detected": res.objects_detected,
            "objects_evaluated": res.objects_evaluated,
            "highest_threat_level": res.highest_threat_level,
            "highest_threat_object_id": res.highest_threat_object_id,
            "total_processing_time_ms": round(total_time_ms, 3),
            "assessments": captured_assessments,
            "errors": captured_errors,
        },
        "disclaimer": SAFETY_DISCLAIMER,
    }

    OUTPUT_DEMO_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DEMO_JSON, "w", encoding="utf-8") as f:
        json.dump(demo_output, f, indent=2)

    print(f"\n[OK] Demonstration results saved to: {OUTPUT_DEMO_JSON}")
    return demo_output


if __name__ == "__main__":
    run_pipeline_demo()
