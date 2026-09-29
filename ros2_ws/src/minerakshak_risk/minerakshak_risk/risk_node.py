"""ROS 2 Perception-to-Risk Assessment Node for MineRakshak.

Consumes detected obstacle clusters from the perception pipeline (/perception/detected_objects),
transforms them into the Stage 8 12-field raw inference contract, runs the frozen
HistGradientBoostingClassifier model, and publishes structured risk assessments
to /risk/assessments and /risk/highest_threat.

Designed with complete fault-isolation: a malformed obstacle observation never crashes
the node or impedes the evaluation of other objects in the frame.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import (
    InferenceValidationError,
    MineRakshakInferenceEngine,
    RESPONSE_MAPPINGS,
    SAFETY_DISCLAIMER,
    get_inference_engine,
)
from src.preprocessing.schema import RISK_CLASSES
from minerakshak_risk.schemas import (
    DetectedObject,
    ObjectRiskAssessment,
    PerceptionFrame,
    RiskAssessmentFrame,
)

# Optional ROS 2 import with safe fallback
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Dummy base class for non-ROS environments
    String = None

# Threat hierarchy for selecting highest risk obstacle
THREAT_PRIORITY = {
    "SAFE": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}


class PerceptionToRiskProcessor:
    """Core perception-to-risk frame processor.

    Decoupled from ROS middleware so it can be rigorously tested, validated,
    and benchmarked in standard Python environments.
    """

    def __init__(
        self,
        engine: MineRakshakInferenceEngine | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.engine = engine or get_inference_engine()
        self.logger = logger or logging.getLogger("PerceptionToRiskProcessor")

    def process_frame(
        self,
        frame_input: PerceptionFrame | dict[str, Any] | str,
        include_secondary: bool = False,
    ) -> RiskAssessmentFrame:
        """Process an entire perception frame of detected obstacles.

        Args:
            frame_input: PerceptionFrame instance, dictionary, or JSON string.
            include_secondary: If True, includes secondary XGBoost verification vote.

        Returns:
            RiskAssessmentFrame containing all evaluated objects, errors, and threat summary.
        """
        t_start = time.perf_counter()

        # Parse frame
        if isinstance(frame_input, str):
            try:
                frame = PerceptionFrame.from_json(frame_input)
            except Exception as e:
                err_msg = f"Failed to parse perception frame JSON: {e}"
                self.logger.error(err_msg)
                return RiskAssessmentFrame(
                    frame_id="invalid_frame",
                    timestamp_ns=0,
                    inference_timestamp=datetime.now(timezone.utc).isoformat(),
                    processing_time_ms=(time.perf_counter() - t_start) * 1000.0,
                    objects_detected=0,
                    objects_evaluated=0,
                    highest_threat_level="SAFE",
                    highest_threat_object_id=None,
                    errors=[{"error": err_msg}],
                )
        elif isinstance(frame_input, dict):
            frame = PerceptionFrame.from_dict(frame_input)
        elif isinstance(frame_input, PerceptionFrame):
            frame = frame_input
        else:
            raise ValueError(f"Unsupported frame input type: {type(frame_input).__name__}")

        total_detected = len(frame.objects)
        assessments: list[ObjectRiskAssessment] = []
        errors: list[dict[str, str]] = []

        # Process each detected object with isolated fault-tolerance
        valid_inference_dicts: list[dict[str, Any]] = []
        valid_objects: list[DetectedObject] = []

        for obj in frame.objects:
            try:
                raw_dict = obj.to_raw_inference_dict(default_truck_speed=frame.truck_speed_kmph)
                cleaned_dict = self.engine.validate_observation(raw_dict)
                valid_inference_dicts.append(cleaned_dict)
                valid_objects.append(obj)
            except (InferenceValidationError, Exception) as val_err:
                self.logger.warning(
                    f"Frame '{frame.frame_id}' object '{obj.object_id}' rejected by validation: {val_err}"
                )
                errors.append({
                    "object_id": obj.object_id,
                    "error": str(val_err),
                })

        # Run batch ML inference on valid objects
        now_iso = datetime.now(timezone.utc).isoformat()
        if valid_inference_dicts:
            t_inf_start = time.perf_counter()
            predictions = self.engine.predict_batch(
                valid_inference_dicts,
                include_secondary=include_secondary,
            )
            batch_inf_time_ms = (time.perf_counter() - t_inf_start) * 1000.0
            per_obj_lat_ms = round(batch_inf_time_ms / len(valid_inference_dicts), 4)

            sensor_source = getattr(frame, "sensor_source", "2D_Planar_LiDAR")
            source_type = getattr(frame, "source_type", "synthetic")
            hardware_validated = getattr(frame, "hardware_validated", False)
            sensor_health = getattr(frame, "sensor_health", "OK")
            feature_compatibility_status = getattr(frame, "feature_compatibility_status", "COMPATIBLE_WITH_DEFAULTS")

            for obj, pred in zip(valid_objects, predictions):
                assessments.append(ObjectRiskAssessment(
                    object_id=obj.object_id,
                    risk_level=pred["risk_level"],
                    class_id=pred["class_id"],
                    confidence=pred["confidence"],
                    probabilities=pred["probabilities"],
                    model=pred["model"],
                    status=pred["status"],
                    recommendation=pred["recommendation"],
                    predicted_risk_level=pred["predicted_risk_level"],
                    risk_code=pred["risk_code"],
                    recommended_action=pred["recommended_action"],
                    primary_model=pred["primary_model"],
                    disclaimer=pred["disclaimer"],
                    inference_latency_ms=per_obj_lat_ms,
                    inference_timestamp=now_iso,
                    sensor_source=sensor_source,
                    source_type=source_type,
                    hardware_validated=hardware_validated,
                    sensor_health=sensor_health,
                    feature_compatibility_status=feature_compatibility_status,
                    distance_m=obj.distance_m,
                    relative_velocity_mps=obj.relative_velocity_mps,
                    time_to_collision_s=obj.time_to_collision_s,
                    object_x_m=obj.object_x_m,
                    object_y_m=obj.object_y_m,
                ))

        # Determine highest threat obstacle
        highest_threat, highest_threat_obj_id = self.determine_highest_threat_with_id(assessments)

        processing_time_ms = (time.perf_counter() - t_start) * 1000.0

        return RiskAssessmentFrame(
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            inference_timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=processing_time_ms,
            objects_detected=total_detected,
            objects_evaluated=len(assessments),
            highest_threat_level=highest_threat,
            highest_threat_object_id=highest_threat_obj_id,
            assessments=assessments,
            errors=errors,
            sensor_source=getattr(frame, "sensor_source", "2D_Planar_LiDAR"),
            source_type=getattr(frame, "source_type", "synthetic"),
            hardware_validated=getattr(frame, "hardware_validated", False),
            sensor_health=getattr(frame, "sensor_health", "OK"),
            feature_compatibility_status=getattr(frame, "feature_compatibility_status", "COMPATIBLE_WITH_DEFAULTS"),
        )

    @classmethod
    def _determine_highest_threat(cls, assessments: list[ObjectRiskAssessment]) -> str:
        """Determine highest threat risk tier across a list of assessments."""
        level, _ = cls.determine_highest_threat_with_id(assessments)
        return level

    @staticmethod
    def determine_highest_threat_with_id(assessments: list[ObjectRiskAssessment]) -> tuple[str, str | None]:
        """Determine highest threat risk tier and corresponding object ID."""
        highest_threat = "SAFE"
        highest_threat_obj_id = None
        max_priority = -1

        for a in assessments:
            pri = THREAT_PRIORITY.get(a.predicted_risk_level, 0)
            if pri > max_priority:
                max_priority = pri
                highest_threat = a.predicted_risk_level
                highest_threat_obj_id = a.object_id

        return highest_threat, highest_threat_obj_id

    def process_observation(
        self,
        observation: dict[str, Any] | str,
        include_secondary: bool = False,
    ) -> dict[str, Any]:
        """Process a single raw obstacle observation with strict safety validation.

        Args:
            observation: Dictionary of physical features or JSON string.
            include_secondary: If True, includes secondary XGBoost verification.

        Returns:
            Dictionary with predicted risk tier, class_id, confidence, probabilities,
            model, status, and safety recommendation.
        """
        if isinstance(observation, str):
            try:
                obs_dict = json.loads(observation)
            except Exception as e:
                err_msg = f"Failed to parse observation JSON: {e}"
                self.logger.error(err_msg)
                return {
                    "risk_level": "INVALID",
                    "class_id": -1,
                    "confidence": 0.0,
                    "probabilities": {"SAFE": 0.0, "CAUTION": 0.0, "WARNING": 0.0, "CRITICAL": 0.0},
                    "model": "HistGradientBoostingClassifier",
                    "status": "error",
                    "error": err_msg,
                    "recommendation": "sensor validation error - do not proceed",
                    "disclaimer": SAFETY_DISCLAIMER,
                }
        else:
            obs_dict = observation

        try:
            return self.engine.predict_single(obs_dict, include_secondary=include_secondary)
        except (InferenceValidationError, ValueError) as val_err:
            self.logger.warning(f"Obstacle observation rejected by validation: {val_err}")
            return {
                "risk_level": "INVALID",
                "class_id": -1,
                "confidence": 0.0,
                "probabilities": {"SAFE": 0.0, "CAUTION": 0.0, "WARNING": 0.0, "CRITICAL": 0.0},
                "model": "HistGradientBoostingClassifier",
                "status": "error",
                "error": str(val_err),
                "recommendation": "sensor validation error - do not proceed",
                "disclaimer": SAFETY_DISCLAIMER,
            }


class MineRakshakRiskNode(Node):
    """ROS 2 Node subscribing to obstacle perception and publishing risk evaluations."""

    def __init__(self, node_name: str = "minerakshak_risk_node") -> None:
        self.node_name = node_name
        self.enable_secondary = False
        self.log_latency = True

        if not ROS2_AVAILABLE:
            self.processor = PerceptionToRiskProcessor()
            return

        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("input_topic", "/minerakshak/object_observations")
        self.declare_parameter("output_topic", "/minerakshak/risk_prediction")
        self.declare_parameter("threat_topic", "/minerakshak/highest_threat")
        self.declare_parameter("perception_topic", "")
        self.declare_parameter("risk_topic", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("preprocessor_path", "")
        self.declare_parameter("feature_schema_path", "")
        self.declare_parameter("label_mapping_path", "")
        self.declare_parameter("enable_secondary_voter", False)
        self.declare_parameter("log_latency", True)

        in_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        perception_topic = self.get_parameter("perception_topic").get_parameter_value().string_value
        actual_input_topic = perception_topic if perception_topic else in_topic

        out_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        risk_topic = self.get_parameter("risk_topic").get_parameter_value().string_value
        actual_output_topic = risk_topic if risk_topic else out_topic

        actual_threat_topic = self.get_parameter("threat_topic").get_parameter_value().string_value
        self.enable_secondary = self.get_parameter("enable_secondary_voter").get_parameter_value().bool_value
        self.log_latency = self.get_parameter("log_latency").get_parameter_value().bool_value

        # Configure model and preprocessor paths (package-relative or configurable)
        model_p = self.get_parameter("model_path").get_parameter_value().string_value
        prep_p = self.get_parameter("preprocessor_path").get_parameter_value().string_value
        schema_p = self.get_parameter("feature_schema_path").get_parameter_value().string_value
        label_p = self.get_parameter("label_mapping_path").get_parameter_value().string_value

        if model_p or prep_p or schema_p or label_p:
            models_dir = PROJECT_ROOT / "models"
            engine = MineRakshakInferenceEngine(
                primary_model_path=Path(model_p) if model_p else models_dir / "hgb_model.joblib",
                preprocessor_path=Path(prep_p) if prep_p else models_dir / "preprocessor.joblib",
                feature_schema_path=Path(schema_p) if schema_p else models_dir / "feature_schema.json",
                label_mapping_path=Path(label_p) if label_p else models_dir / "label_mapping.json",
                load_secondary=self.enable_secondary,
            )
        else:
            engine = get_inference_engine()

        self.processor = PerceptionToRiskProcessor(engine=engine, logger=self.get_logger())

        # ROS 2 Publisher and Subscriber
        self.subscription = self.create_subscription(
            String,
            actual_input_topic,
            self.listener_callback,
            10,
        )
        self.risk_publisher = self.create_publisher(String, actual_output_topic, 10)
        self.threat_publisher = self.create_publisher(String, actual_threat_topic, 10)

        self.get_logger().info(
            f"MineRakshak Risk Node started. Subscribed to '{actual_input_topic}', publishing to '{actual_output_topic}'."
        )

    def listener_callback(self, msg: Any) -> Any:
        """Callback invoked when an observation or perception frame is received."""
        raw_text = msg.data if hasattr(msg, "data") else str(msg)

        # Detect whether the payload is a single observation or a perception frame
        is_single_obs = False
        parsed_data = None
        try:
            parsed_data = json.loads(raw_text)
            if isinstance(parsed_data, dict) and "distance_m" in parsed_data and "objects" not in parsed_data:
                is_single_obs = True
        except Exception:
            pass

        # 1. Single obstacle observation path
        if is_single_obs and parsed_data is not None:
            pred_result = self.processor.process_observation(parsed_data, include_secondary=self.enable_secondary)
            if ROS2_AVAILABLE and hasattr(self, "risk_publisher") and self.risk_publisher is not None:
                out_msg = String()
                out_msg.data = json.dumps(pred_result)
                self.risk_publisher.publish(out_msg)

                if self.log_latency:
                    self.get_logger().info(
                        f"Obstacle inference: Risk={pred_result.get('risk_level', 'UNKNOWN')}, "
                        f"Confidence={pred_result.get('confidence', 0.0):.4f}"
                    )
            return pred_result

        # 2. Multi-obstacle perception frame path
        assessment_frame = self.processor.process_frame(raw_text, include_secondary=self.enable_secondary)

        if ROS2_AVAILABLE and hasattr(self, "risk_publisher") and self.risk_publisher is not None:
            out_msg = String()
            out_msg.data = assessment_frame.to_json()
            self.risk_publisher.publish(out_msg)

            # Publish highest threat summary
            threat_msg = String()
            threat_msg.data = json.dumps({
                "frame_id": assessment_frame.frame_id,
                "highest_threat_level": assessment_frame.highest_threat_level,
                "highest_threat_object_id": assessment_frame.highest_threat_object_id,
                "action": RESPONSE_MAPPINGS.get(assessment_frame.highest_threat_level, "monitor"),
                "timestamp": assessment_frame.inference_timestamp,
            })
            self.threat_publisher.publish(threat_msg)

            if self.log_latency:
                self.get_logger().info(
                    f"Frame {assessment_frame.frame_id}: {assessment_frame.objects_evaluated}/"
                    f"{assessment_frame.objects_detected} objects evaluated in {assessment_frame.processing_time_ms:.2f} ms. "
                    f"Threat: {assessment_frame.highest_threat_level}"
                )

        return assessment_frame

    def destroy_node(self) -> bool:
        """Cleanly destroy node resources."""
        if ROS2_AVAILABLE and hasattr(super(), "destroy_node"):
            return super().destroy_node()
        self.processor = None
        return True


def main(args: list[str] | None = None) -> None:
    """ROS 2 entry point."""
    if not ROS2_AVAILABLE:
        print("[ERROR] ROS 2 (rclpy) is not installed in the current Python environment.")
        print("To execute this node inside a ROS 2 workspace:")
        print("  1. Ensure ROS 2 (Humble / Iron / Rolling) is installed.")
        print("  2. In your workspace: colcon build --packages-select minerakshak_risk")
        print("  3. source install/setup.bash (or call install/setup.bat)")
        print("  4. ros2 run minerakshak_risk risk_node")
        sys.exit(1)

    rclpy.init(args=args)
    node = MineRakshakRiskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
