"""Perception and Risk Assessment message schemas for MineRakshak ROS 2 node."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DetectedObject:
    """Represents a single detected obstacle cluster from LiDAR perception."""
    object_id: str = "obj_001"
    distance_m: float | None = None
    object_x_m: float | None = None
    object_y_m: float | None = None
    object_z_m: float | None = None
    object_width_m: float | None = None
    object_height_m: float | None = None
    object_length_m: float | None = None
    point_count: int | None = None
    relative_velocity_mps: float | None = None
    object_type: str | None = None
    truck_speed_kmph: float | None = None
    time_to_collision_s: float | None = None

    def to_raw_inference_dict(self, default_truck_speed: float = 0.0) -> dict[str, Any]:
        """Convert into Stage 8 MineRakshak 12-feature raw inference dictionary."""
        speed = self.truck_speed_kmph if self.truck_speed_kmph is not None else default_truck_speed
        d: dict[str, Any] = {}
        for key in [
            "distance_m", "object_x_m", "object_y_m", "object_z_m",
            "object_width_m", "object_height_m", "object_length_m",
            "point_count", "relative_velocity_mps"
        ]:
            val = getattr(self, key, None)
            if val is not None:
                d[key] = val
        if speed is not None:
            d["truck_speed_kmph"] = speed
        if self.object_type is not None:
            d["object_type"] = self.object_type
        if self.time_to_collision_s is not None:
            d["time_to_collision_s"] = self.time_to_collision_s
        return d


@dataclass
class PerceptionFrame:
    """Represents a perception cycle frame containing detected obstacles."""
    frame_id: str
    timestamp_ns: int
    objects: list[DetectedObject] = field(default_factory=list)
    truck_speed_kmph: float = 0.0
    sensor_source: str = "2D_Planar_LiDAR"
    source_type: str = "synthetic"  # synthetic, recorded, or real_ros2
    hardware_validated: bool = False
    sensor_health: str = "OK"
    feature_compatibility_status: str = "COMPATIBLE_WITH_DEFAULTS"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerceptionFrame:
        """Parse from dictionary."""
        frame_id = str(data.get("frame_id", "lidar_frame"))
        timestamp_ns = int(data.get("timestamp_ns", 0))
        truck_speed = float(data.get("truck_speed_kmph", 0.0))
        sensor_source = str(data.get("sensor_source", "2D_Planar_LiDAR"))
        source_type = str(data.get("source_type", "synthetic"))
        hardware_validated = bool(data.get("hardware_validated", False))
        sensor_health = str(data.get("sensor_health", "OK"))
        feature_compatibility_status = str(data.get("feature_compatibility_status", "COMPATIBLE_WITH_DEFAULTS"))

        raw_objects = data.get("objects", [])
        parsed_objects: list[DetectedObject] = []

        for idx, obj in enumerate(raw_objects):
            if not isinstance(obj, dict):
                continue
            obj_id = str(obj.get("object_id", f"obj_{idx:03d}"))
            parsed_objects.append(DetectedObject(
                object_id=obj_id,
                distance_m=obj.get("distance_m"),
                object_x_m=obj.get("object_x_m"),
                object_y_m=obj.get("object_y_m"),
                object_z_m=obj.get("object_z_m"),
                object_width_m=obj.get("object_width_m"),
                object_height_m=obj.get("object_height_m"),
                object_length_m=obj.get("object_length_m"),
                point_count=obj.get("point_count"),
                relative_velocity_mps=obj.get("relative_velocity_mps"),
                object_type=obj.get("object_type"),
                truck_speed_kmph=obj.get("truck_speed_kmph"),
                time_to_collision_s=obj.get("time_to_collision_s"),
            ))

        return cls(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            objects=parsed_objects,
            truck_speed_kmph=truck_speed,
            sensor_source=sensor_source,
            source_type=source_type,
            hardware_validated=hardware_validated,
            sensor_health=sensor_health,
            feature_compatibility_status=feature_compatibility_status,
        )

    @classmethod
    def from_json(cls, json_str: str) -> PerceptionFrame:
        """Parse from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> dict[str, Any]:
        """Convert into dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "truck_speed_kmph": self.truck_speed_kmph,
            "sensor_source": self.sensor_source,
            "source_type": self.source_type,
            "hardware_validated": self.hardware_validated,
            "sensor_health": self.sensor_health,
            "feature_compatibility_status": self.feature_compatibility_status,
            "objects": [asdict(obj) for obj in self.objects],
        }

    def to_json(self) -> str:
        """Convert into JSON string."""
        return json.dumps(self.to_dict())



@dataclass
class ObjectRiskAssessment:
    """Individual object risk prediction output complying with Stage 8 contract."""
    object_id: str
    risk_level: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    model: str = "HistGradientBoostingClassifier"
    status: str = "valid"
    recommendation: str = "no immediate hazard"
    predicted_risk_level: str = ""
    risk_code: int = 0
    recommended_action: str = ""
    primary_model: str = ""
    disclaimer: str = ""
    inference_latency_ms: float = 0.0
    inference_timestamp: str = ""
    sensor_source: str = "2D_Planar_LiDAR"
    source_type: str = "synthetic"  # synthetic, recorded, or real_ros2
    hardware_validated: bool = False
    sensor_health: str = "OK"
    feature_compatibility_status: str = "COMPATIBLE_WITH_DEFAULTS"
    distance_m: float | None = None
    relative_velocity_mps: float | None = None
    time_to_collision_s: float | None = None
    object_x_m: float | None = None
    object_y_m: float | None = None

    def __post_init__(self) -> None:
        if not self.predicted_risk_level:
            self.predicted_risk_level = self.risk_level
        if not self.risk_code:
            self.risk_code = self.class_id
        if not self.recommended_action:
            self.recommended_action = self.recommendation
        if not self.primary_model:
            self.primary_model = self.model

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "risk_level": self.risk_level,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "probabilities": self.probabilities,
            "model": self.model,
            "status": self.status,
            "recommendation": self.recommendation,
            "predicted_risk_level": self.predicted_risk_level,
            "risk_code": self.risk_code,
            "recommended_action": self.recommended_action,
            "primary_model": self.primary_model,
            "disclaimer": self.disclaimer,
            "inference_latency_ms": round(self.inference_latency_ms, 3),
            "inference_timestamp": self.inference_timestamp,
            "sensor_source": self.sensor_source,
            "source_type": self.source_type,
            "hardware_validated": self.hardware_validated,
            "sensor_health": self.sensor_health,
            "feature_compatibility_status": self.feature_compatibility_status,
            "distance_m": round(self.distance_m, 3) if self.distance_m is not None else None,
            "relative_velocity_mps": round(self.relative_velocity_mps, 3) if self.relative_velocity_mps is not None else None,
            "time_to_collision_s": round(self.time_to_collision_s, 2) if self.time_to_collision_s is not None else None,
            "object_x_m": round(self.object_x_m, 3) if self.object_x_m is not None else None,
            "object_y_m": round(self.object_y_m, 3) if self.object_y_m is not None else None,
        }


@dataclass
class RiskAssessmentFrame:
    """Complete published risk assessment frame for all objects in a perception cycle."""
    frame_id: str
    timestamp_ns: int
    inference_timestamp: str
    processing_time_ms: float
    objects_detected: int
    objects_evaluated: int
    highest_threat_level: str
    highest_threat_object_id: str | None
    assessments: list[ObjectRiskAssessment] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    sensor_source: str = "2D_Planar_LiDAR"
    source_type: str = "synthetic"  # synthetic, recorded, or real_ros2
    hardware_validated: bool = False
    sensor_health: str = "OK"
    feature_compatibility_status: str = "COMPATIBLE_WITH_DEFAULTS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "inference_timestamp": self.inference_timestamp,
            "processing_time_ms": round(self.processing_time_ms, 3),
            "objects_detected": self.objects_detected,
            "objects_evaluated": self.objects_evaluated,
            "highest_threat_level": self.highest_threat_level,
            "highest_threat_object_id": self.highest_threat_object_id,
            "assessments": [a.to_dict() for a in self.assessments],
            "errors": self.errors,
            "sensor_source": self.sensor_source,
            "source_type": self.source_type,
            "hardware_validated": self.hardware_validated,
            "sensor_health": self.sensor_health,
            "feature_compatibility_status": self.feature_compatibility_status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
