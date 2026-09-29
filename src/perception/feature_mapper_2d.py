"""Stage 12: 2D-to-Risk Model Feature Compatibility & Mapping Layer.

===============================================================================
MANDATORY ARCHITECTURAL REQUIREMENT:
MODEL COMPATIBILITY REQUIRES REAL-SENSOR VALIDATION

The existing HistGradientBoostingClassifier and MineRakshakPreprocessor were trained
in earlier stages on a 12-feature raw schema (representing 3D LiDAR bounding boxes).
The actual target hardware for this project is a 2D planar LiDAR.

A 2D LiDAR CANNOT directly provide:
  - object elevation / z-coordinate (object_z_m)
  - vertical object height (object_height_m)
  - true 3D bounding boxes
  - single-scan object classification (object_type)

This module provides a strictly honest, auditable compatibility mapping layer:
1. Available 2D features (distance, x, y, planar width, depth, beam count, tracker velocity)
   are mapped directly.
2. Missing 3D features (z, height, object_type) are NOT fabricated as real measurements.
   They are explicitly mapped to neutral nominal baselines to satisfy the frozen
   inference engine's input contract, with every limitation documented in metadata.
3. Every mapped output carries explicit audit flags declaring which features are genuine
   2D planar observations and which are contract-compatibility defaults.
===============================================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.perception.lidar_2d import Obstacle2D

# Explicit compatibility statement required by Stage 12 specifications
STAGE12_COMPATIBILITY_STATEMENT = "MODEL COMPATIBILITY REQUIRES REAL-SENSOR VALIDATION"

# Feature availability definitions
FEATURE_COMPATIBILITY_MATRIX = [
    {
        "feature_name": "distance_m",
        "availability_2d": "Available",
        "derivation_method": "Euclidean range to 2D cluster centroid: sqrt(x^2 + y^2)",
        "limitation": "Measures distance to nearest planar cross-section, not 3D centroid.",
    },
    {
        "feature_name": "object_x_m",
        "availability_2d": "Available",
        "derivation_method": "Planar longitudinal coordinate (+X forward in vehicle frame)",
        "limitation": "Planar forward distance only.",
    },
    {
        "feature_name": "object_y_m",
        "availability_2d": "Available",
        "derivation_method": "Planar lateral coordinate (+/-Y lateral in vehicle frame)",
        "limitation": "Planar lateral distance only.",
    },
    {
        "feature_name": "object_z_m",
        "availability_2d": "NOT directly available",
        "derivation_method": "Contract default (0.0 m nominal sensor mount plane)",
        "limitation": "2D LiDAR has single scan plane; cannot measure elevation. Do not fabricate.",
    },
    {
        "feature_name": "object_width_m",
        "availability_2d": "Approximate",
        "derivation_method": "Planar cluster span across beam extremities in scan plane",
        "limitation": "Visible cross-section width only; occluded edges cannot be seen.",
    },
    {
        "feature_name": "object_height_m",
        "availability_2d": "NOT available",
        "derivation_method": "Contract default (1.5 m nominal baseline)",
        "limitation": "2D LiDAR cannot measure vertical extent. Do not fabricate.",
    },
    {
        "feature_name": "object_length_m",
        "availability_2d": "Approximate/Conditional",
        "derivation_method": "Radial beam depth span along line of sight",
        "limitation": "Trailing geometry of obstacle is occluded in planar scan.",
    },
    {
        "feature_name": "point_count",
        "availability_2d": "Available",
        "derivation_method": "Number of beam returns clustered into 2D obstacle",
        "limitation": "Represents 2D planar beam count, not 3D point cloud count.",
    },
    {
        "feature_name": "relative_velocity_mps",
        "availability_2d": "Multi-scan derived",
        "derivation_method": "Temporal tracking across consecutive scans: delta_d / delta_t",
        "limitation": "Requires at least 2 consecutive cycles; single scan cannot measure velocity.",
    },
    {
        "feature_name": "truck_speed_kmph",
        "availability_2d": "External source",
        "derivation_method": "Vehicle CAN bus / GPS / odometry telemetry",
        "limitation": "Not directly measured by LiDAR sensor.",
    },
    {
        "feature_name": "time_to_collision_s",
        "availability_2d": "Derived",
        "derivation_method": "Kinematic calculation: distance / (-relative_velocity) if approaching",
        "limitation": "Inherits multi-scan tracking velocity estimation uncertainties.",
    },
    {
        "feature_name": "object_type",
        "availability_2d": "NOT reliably available",
        "derivation_method": "Default 'unknown' (or external vision/camera fusion tag)",
        "limitation": "Single 2D slice contour cannot distinguish car, truck, crane, or person.",
    },
]


@dataclass
class FeatureCompatibilityMetadata:
    """Explicit audit record accompanying 2D-mapped inference observations."""

    obstacle_id: str
    compatibility_statement: str = STAGE12_COMPATIBILITY_STATEMENT
    sensor_type: str = "2D_PLANAR_LIDAR"
    z_is_real_measurement: bool = False
    height_is_real_measurement: bool = False
    classification_is_real_measurement: bool = False
    velocity_source: str = "MULTI_SCAN_TRACKER"
    z_value_used: float = 0.0
    height_value_used: float = 1.5
    object_type_used: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureMapper2D:
    """Maps 2D planar obstacles to the frozen 12-feature raw ML inference schema."""

    def __init__(
        self,
        nominal_z_m: float = 0.0,
        nominal_height_m: float = 1.5,
        default_object_type: str = "unknown",
        default_truck_speed_kmph: float = 25.0,
    ) -> None:
        """Initialize feature mapper.

        Args:
            nominal_z_m: Nominal elevation value used solely to satisfy the frozen preprocessor contract.
            nominal_height_m: Nominal height value used solely to satisfy the frozen preprocessor contract.
            default_object_type: Default category when no external perception/camera fusion is provided.
            default_truck_speed_kmph: Default truck speed if external vehicle odometry is absent.
        """
        self.nominal_z = nominal_z_m
        self.nominal_height = nominal_height_m
        self.default_object_type = default_object_type
        self.default_truck_speed = default_truck_speed_kmph

    def map_obstacle(
        self,
        obstacle: Obstacle2D,
        truck_speed_kmph: float | None = None,
        external_object_type: str | None = None,
    ) -> tuple[dict[str, Any], FeatureCompatibilityMetadata]:
        """Convert a 2D obstacle into the 12-feature dictionary required by frozen HGB inference.

        Returns:
            Tuple of (raw_inference_dict, audit_metadata).
        """
        speed = truck_speed_kmph if truck_speed_kmph is not None else self.default_truck_speed
        obj_type = external_object_type if external_object_type is not None else self.default_object_type

        # Construct strictly compliant 12-feature raw dictionary
        inference_dict: dict[str, Any] = {
            "distance_m": float(obstacle.distance_m),
            "object_x_m": float(obstacle.object_x_m),
            "object_y_m": float(obstacle.object_y_m),
            "object_z_m": float(self.nominal_z),  # Explicit nominal plane
            "object_width_m": float(max(0.2, obstacle.span_width_m)),
            "object_height_m": float(self.nominal_height),  # Explicit nominal baseline
            "object_length_m": float(max(0.2, obstacle.depth_length_m)),
            "point_count": int(max(1, obstacle.point_count)),
            "relative_velocity_mps": float(obstacle.relative_velocity_mps),
            "truck_speed_kmph": float(speed),
            "time_to_collision_s": float(obstacle.time_to_collision_s),
            "object_type": str(obj_type),
        }

        # Build transparent audit record
        metadata = FeatureCompatibilityMetadata(
            obstacle_id=obstacle.obstacle_id,
            compatibility_statement=STAGE12_COMPATIBILITY_STATEMENT,
            sensor_type="2D_PLANAR_LIDAR",
            z_is_real_measurement=False,
            height_is_real_measurement=False,
            classification_is_real_measurement=(external_object_type is not None),
            velocity_source="MULTI_SCAN_TRACKER" if obstacle.relative_velocity_mps != 0.0 else "UNINITIALIZED_SINGLE_CYCLE",
            z_value_used=self.nominal_z,
            height_value_used=self.nominal_height,
            object_type_used=obj_type,
        )

        return inference_dict, metadata

    def map_obstacles_batch(
        self,
        obstacles: Sequence[Obstacle2D],
        truck_speed_kmph: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[FeatureCompatibilityMetadata]]:
        """Batch map a sequence of 2D obstacles."""
        dicts: list[dict[str, Any]] = []
        metas: list[FeatureCompatibilityMetadata] = []

        for obs in obstacles:
            inf_d, meta = self.map_obstacle(obs, truck_speed_kmph=truck_speed_kmph)
            dicts.append(inf_d)
            metas.append(meta)

        return dicts, metas

    def to_detected_objects(
        self,
        obstacles: Sequence[Obstacle2D],
        truck_speed_kmph: float | None = None,
        external_object_type: str | None = None,
    ) -> list[Any]:
        """Convert 2D obstacles to ROS 2 DetectedObject schemas for downstream node compatibility."""
        from minerakshak_risk.schemas import DetectedObject

        detected_objects: list[DetectedObject] = []
        for obs in obstacles:
            inf_d, _ = self.map_obstacle(
                obs,
                truck_speed_kmph=truck_speed_kmph,
                external_object_type=external_object_type,
            )
            detected_objects.append(
                DetectedObject(
                    object_id=obs.obstacle_id,
                    distance_m=inf_d["distance_m"],
                    object_x_m=inf_d["object_x_m"],
                    object_y_m=inf_d["object_y_m"],
                    object_z_m=inf_d["object_z_m"],
                    object_width_m=inf_d["object_width_m"],
                    object_height_m=inf_d["object_height_m"],
                    object_length_m=inf_d["object_length_m"],
                    point_count=inf_d["point_count"],
                    relative_velocity_mps=inf_d["relative_velocity_mps"],
                    object_type=inf_d["object_type"],
                    truck_speed_kmph=inf_d["truck_speed_kmph"],
                    time_to_collision_s=inf_d["time_to_collision_s"],
                )
            )
        return detected_objects

    def to_perception_frame(
        self,
        obstacles: Sequence[Obstacle2D],
        frame_id: str,
        timestamp_ns: int,
        truck_speed_kmph: float = 25.0,
        sensor_source: str = "2D_Planar_LiDAR",
        source_type: str = "synthetic",
        hardware_validated: bool = False,
        sensor_health: str = "OK",
        feature_compatibility_status: str = "COMPATIBLE_WITH_DEFAULTS",
    ) -> Any:
        """Assemble a ROS 2-compatible PerceptionFrame from 2D obstacle detections."""
        from minerakshak_risk.schemas import PerceptionFrame

        detected_objects = self.to_detected_objects(obstacles, truck_speed_kmph=truck_speed_kmph)
        return PerceptionFrame(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            objects=detected_objects,
            truck_speed_kmph=truck_speed_kmph,
            sensor_source=sensor_source,
            source_type=source_type,
            hardware_validated=hardware_validated,
            sensor_health=sensor_health,
            feature_compatibility_status=feature_compatibility_status,
        )

    def save_compatibility_matrix_csv(self, output_path: Path | str) -> None:
        """Export feature compatibility matrix to CSV."""
        import csv

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["feature_name", "availability_2d", "derivation_method", "limitation"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in FEATURE_COMPATIBILITY_MATRIX:
                writer.writerow(row)
