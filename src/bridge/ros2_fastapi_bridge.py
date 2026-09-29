"""Stage 11: ROS 2 to FastAPI Telemetry Bridge.

Connects the full pipeline:
3D LiDAR / perception adapter
  -> ROS 2 /minerakshak/object_observations
  -> Frozen HGB inference (PerceptionToRiskProcessor)
  -> ROS 2 /minerakshak/risk_prediction + /minerakshak/highest_threat
  -> FastAPI backend telemetry endpoints
  -> Dashboard WebSocket broadcast.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Ensure project root, parent directory, and ROS 2 package are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT), str(ROS2_PKG_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

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
from src.inference.predict import (
    RESPONSE_MAPPINGS,
    SAFETY_DISCLAIMER,
    get_inference_engine,
)
from src.perception.mock_lidar_adapter import MockLiDARPerceptionAdapter
from minerakshak_risk.laserscan_node import LaserScanToRiskPipeline

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import main as fastapi_app_module
    FASTAPI_MODULE_AVAILABLE = True
except ImportError:
    fastapi_app_module = None
    FASTAPI_MODULE_AVAILABLE = False

logger = logging.getLogger("MineRakshakBridge")


class MineRakshakSystemBridge:
    """Full-system integration bridge coordinating Perception -> ROS 2 -> ML -> FastAPI -> WebSockets."""

    def __init__(
        self,
        fastapi_base_url: str | None = None,
        use_in_process_fastapi: bool = True,
        processor: PerceptionToRiskProcessor | None = None,
    ) -> None:
        self.fastapi_base_url = fastapi_base_url
        self.use_in_process_fastapi = use_in_process_fastapi
        self.processor = processor or PerceptionToRiskProcessor()
        self.node = MineRakshakRiskNode()
        self.perception_adapter = MockLiDARPerceptionAdapter()
        self.laserscan_pipeline = LaserScanToRiskPipeline(risk_processor=self.processor)

        # Telemetry statistics
        self.total_frames_processed = 0
        self.total_objects_evaluated = 0
        self.total_errors_isolated = 0
        self.last_risk_frame: RiskAssessmentFrame | None = None
        self.last_threat_summary: dict[str, Any] | None = None

        logger.info(
            f"MineRakshakSystemBridge initialized. ROS2 Available: {ROS2_AVAILABLE}, "
            f"FastAPI In-Process: {self.use_in_process_fastapi}"
        )

    def process_perception_to_dashboard(
        self,
        frame_input: PerceptionFrame | dict[str, Any] | str,
        include_secondary: bool = False,
    ) -> dict[str, Any]:
        """Execute the complete end-to-end software pipeline synchronously or in-process.

        Stage breakdown:
        1. Ingest from perception adapter
        2. Publish/route to ROS 2 object_observations
        3. Validate input & run frozen HGB inference
        4. Emit ROS 2 risk_prediction and highest_threat payloads
        5. Forward risk outputs to FastAPI backend
        6. Broadcast to dashboard and dedicated WebSockets.

        Returns:
            Dictionary containing evaluation results, stage timings, and telemetry payload.
        """
        t_pipeline_start = time.perf_counter()

        # 1. Perception formatting / serialization
        t_percep_start = time.perf_counter()
        if isinstance(frame_input, PerceptionFrame):
            raw_frame_json = frame_input.to_json()
            pframe = frame_input
        elif isinstance(frame_input, dict):
            pframe = PerceptionFrame.from_dict(frame_input)
            raw_frame_json = json.dumps(frame_input)
        elif isinstance(frame_input, str):
            try:
                pframe = PerceptionFrame.from_json(frame_input)
            except Exception:
                pframe = None
            raw_frame_json = frame_input
        else:
            raise ValueError(f"Unsupported frame input type: {type(frame_input).__name__}")
        t_perception_ms = (time.perf_counter() - t_percep_start) * 1000.0

        # 2. ROS 2 Ingestion & ML Inference
        t_ros2_start = time.perf_counter()
        if pframe is not None:
            assessment_frame: RiskAssessmentFrame = self.processor.process_frame(
                pframe, include_secondary=include_secondary
            )
        else:
            # Handle corrupt/malformed raw input string
            assessment_frame = self.processor.process_frame(
                raw_frame_json, include_secondary=include_secondary
            )
        t_ros2_ml_ms = (time.perf_counter() - t_ros2_start) * 1000.0

        self.last_risk_frame = assessment_frame
        self.total_frames_processed += 1
        self.total_objects_evaluated += assessment_frame.objects_evaluated
        self.total_errors_isolated += len(assessment_frame.errors)

        # 3. ROS 2 Published Payload Serialization
        t_ros2_pub_start = time.perf_counter()
        risk_prediction_payload = assessment_frame.to_dict()
        highest_threat_payload = {
            "frame_id": assessment_frame.frame_id,
            "highest_threat_level": assessment_frame.highest_threat_level,
            "highest_threat_object_id": assessment_frame.highest_threat_object_id,
            "action": RESPONSE_MAPPINGS.get(assessment_frame.highest_threat_level, "no immediate hazard"),
            "timestamp": assessment_frame.inference_timestamp,
        }
        self.last_threat_summary = highest_threat_payload
        t_ros2_pub_ms = (time.perf_counter() - t_ros2_pub_start) * 1000.0

        # 4. Bridge to FastAPI Backend
        t_fastapi_start = time.perf_counter()
        fastapi_status = self._forward_to_fastapi(risk_prediction_payload, highest_threat_payload)
        t_fastapi_ms = (time.perf_counter() - t_fastapi_start) * 1000.0

        t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        return {
            "frame_id": assessment_frame.frame_id,
            "highest_threat_level": assessment_frame.highest_threat_level,
            "highest_threat_object_id": assessment_frame.highest_threat_object_id,
            "objects_detected": assessment_frame.objects_detected,
            "objects_evaluated": assessment_frame.objects_evaluated,
            "errors": assessment_frame.errors,
            "assessments": [a.to_dict() for a in assessment_frame.assessments],
            "timings_ms": {
                "perception_ms": round(t_perception_ms, 3),
                "ros2_ml_inference_ms": round(t_ros2_ml_ms, 3),
                "ros2_publish_serialization_ms": round(t_ros2_pub_ms, 3),
                "fastapi_forwarding_ms": round(t_fastapi_ms, 3),
                "total_e2e_ms": round(t_total_ms, 3),
            },
            "fastapi_forward_status": fastapi_status,
            "status": "valid" if not assessment_frame.errors else "partial_fault_isolated",
            "disclaimer": SAFETY_DISCLAIMER,
        }

    def process_laserscan_to_dashboard(
        self,
        scan_msg: Any,
        truck_speed_kmph: float = 25.0,
        current_time_s: float | None = None,
        include_secondary: bool = False,
    ) -> dict[str, Any]:
        """Execute complete end-to-end software pipeline from 2D LaserScan to dashboard.

        Flow:
        LaserScan (msg/dict/2D)
          -> Adapter validation & conversion
          -> 2D Obstacle extraction
          -> Temporal tracking
          -> Feature mapping (to frozen contract)
          -> Frozen HGB inference
          -> ROS 2 payload formatting
          -> FastAPI backend telemetry update
          -> WebSocket broadcast
        """
        t_pipeline_start = time.perf_counter()

        t_proc_start = time.perf_counter()
        val_res, assessment_frame = self.laserscan_pipeline.process_scan(
            scan_msg,
            truck_speed_kmph=truck_speed_kmph,
            current_time_s=current_time_s,
            include_secondary=include_secondary,
        )
        t_proc_ms = (time.perf_counter() - t_proc_start) * 1000.0

        risk_prediction_payload = assessment_frame.to_dict()
        highest_threat_payload = {
            "frame_id": assessment_frame.frame_id,
            "highest_threat_level": assessment_frame.highest_threat_level,
            "highest_threat_object_id": assessment_frame.highest_threat_object_id,
            "action": RESPONSE_MAPPINGS.get(assessment_frame.highest_threat_level, "no immediate hazard"),
            "timestamp": assessment_frame.inference_timestamp,
        }

        t_fastapi_start = time.perf_counter()
        fastapi_status = self._forward_to_fastapi(risk_prediction_payload, highest_threat_payload)
        t_fastapi_ms = (time.perf_counter() - t_fastapi_start) * 1000.0
        t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        self.total_frames_processed += 1
        self.total_objects_evaluated += assessment_frame.objects_evaluated
        if assessment_frame.errors:
            self.total_errors_isolated += len(assessment_frame.errors)
        self.last_risk_frame = assessment_frame
        self.last_threat_summary = highest_threat_payload

        return {
            "frame_id": assessment_frame.frame_id,
            "sensor_health": val_res.health_state.value,
            "sensor_validation": val_res.to_dict(),
            "highest_threat_level": assessment_frame.highest_threat_level,
            "highest_threat_object_id": assessment_frame.highest_threat_object_id,
            "action": highest_threat_payload["action"],
            "objects_detected": assessment_frame.objects_detected,
            "objects_evaluated": assessment_frame.objects_evaluated,
            "assessments": risk_prediction_payload["assessments"],
            "errors": risk_prediction_payload["errors"],
            "timings_ms": {
                "laserscan_processing_ms": round(t_proc_ms, 3),
                "fastapi_forwarding_ms": round(t_fastapi_ms, 3),
                "total_e2e_ms": round(t_total_ms, 3),
            },
            "fastapi_forward_status": fastapi_status,
            "status": "valid" if (val_res.is_valid and not assessment_frame.errors) else "fault_isolated",
            "disclaimer": SAFETY_DISCLAIMER,
        }

    def _forward_to_fastapi(
        self,
        risk_prediction: dict[str, Any],
        highest_threat: dict[str, Any],
    ) -> str:
        """Forward risk payloads to the FastAPI backend via in-process dispatch or HTTP."""
        # 1. In-process dispatch if available
        if self.use_in_process_fastapi and FASTAPI_MODULE_AVAILABLE and fastapi_app_module is not None:
            try:
                # Update backend memory state directly
                fastapi_app_module.LATEST_MINERAKSHAK_RISK = {
                    "frame_id": risk_prediction.get("frame_id", "lidar_frame"),
                    "timestamp_ns": risk_prediction.get("timestamp_ns", 0),
                    "inference_timestamp": risk_prediction.get("inference_timestamp", ""),
                    "highest_threat_level": risk_prediction.get("highest_threat_level", "SAFE"),
                    "highest_threat_object_id": risk_prediction.get("highest_threat_object_id"),
                    "action": highest_threat.get("action", "no immediate hazard"),
                    "recommended_action": highest_threat.get("action", "no immediate hazard"),
                    "objects_detected": risk_prediction.get("objects_detected", 0),
                    "objects_evaluated": risk_prediction.get("objects_evaluated", 0),
                    "processing_time_ms": risk_prediction.get("processing_time_ms", 0.0),
                    "assessments": risk_prediction.get("assessments", []),
                    "errors": risk_prediction.get("errors", []),
                    "status": "valid" if not risk_prediction.get("errors") else "partial_fault_isolated",
                    "model": "HistGradientBoostingClassifier",
                    "disclaimer": SAFETY_DISCLAIMER,
                    "sensor_source": risk_prediction.get("sensor_source", "2D_Planar_LiDAR"),
                    "source_type": risk_prediction.get("source_type", "synthetic"),
                    "hardware_validated": risk_prediction.get("hardware_validated", False),
                    "sensor_health": risk_prediction.get("sensor_health", "OK"),
                    "feature_compatibility_status": risk_prediction.get("feature_compatibility_status", "COMPATIBLE_WITH_DEFAULTS"),
                }

                # Broadcast to active WebSockets if an event loop is running
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(fastapi_app_module.publish_state())
                        loop.create_task(fastapi_app_module.manager.broadcast({
                            "type": "minerakshak_risk_update",
                            "data": fastapi_app_module.LATEST_MINERAKSHAK_RISK,
                        }))
                except Exception:
                    pass

                return "in_process_dispatched"
            except Exception as e:
                logger.warning(f"In-process FastAPI dispatch error: {e}")
                return f"in_process_error: {e}"

        # 2. HTTP forwarding if base URL configured
        if self.fastapi_base_url and HTTPX_AVAILABLE:
            try:
                with httpx.Client(timeout=1.5) as client:
                    client.post(
                        f"{self.fastapi_base_url}/api/minerakshak/risk_prediction",
                        json=risk_prediction,
                    )
                    client.post(
                        f"{self.fastapi_base_url}/api/minerakshak/highest_threat",
                        json=highest_threat,
                    )
                return "http_forwarded"
            except Exception as e:
                logger.warning(f"HTTP forwarding to FastAPI failed ({self.fastapi_base_url}): {e}")
                return f"http_forward_error: {e}"

        return "skipped_no_transport"


_GLOBAL_BRIDGE: MineRakshakSystemBridge | None = None


def get_system_bridge() -> MineRakshakSystemBridge:
    """Return singleton instance of MineRakshakSystemBridge."""
    global _GLOBAL_BRIDGE
    if _GLOBAL_BRIDGE is None:
        _GLOBAL_BRIDGE = MineRakshakSystemBridge()
    return _GLOBAL_BRIDGE
