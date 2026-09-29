"""Official Feature Schema & Safety Conventions for mineRakshak-ai ML Prototype.

===============================================================================
IMPORTANT ARCHITECTURAL DISCLAIMER & ASSUMPTIONS:
===============================================================================
This schema represents the ASSUMED structured feature contract between upstream
sensor processing (e.g., LiDAR point-cloud clustering & bounding-box estimation)
and the downstream AI/ML risk assessment models.

We currently do NOT have physical access to the live ROS 2 LiDAR data stream.
Therefore:
1. These fields are engineering assumptions designed to mirror standard 3D LiDAR
   object detection outputs (e.g. from ROS 2 PointCloud2 clustering pipelines).
2. The coordinate frame, field names, and dimensions will be adapted when
   interfacing with real ROS 2 perception nodes in future stages without altering
   the core inference architecture.
===============================================================================

COORDINATE FRAME CONVENTION (ASSUMED):
-------------------------------------------------------------------------------
Origin: Front bumper / LiDAR sensor mount on the haul truck.
- X-axis (+X): Forward direction of travel (along the truck's heading).
- Y-axis (+Y): Lateral direction (left/right relative to truck centerline).
- Z-axis (+Z): Vertical direction (elevation above ground plane).

RELATIVE VELOCITY CONVENTION:
-------------------------------------------------------------------------------
- Negative (< 0): Object is CLOSING IN (approaching the truck). Higher closing speed
                  results in more negative values (e.g., -12.5 m/s).
- Positive (> 0): Object is RECEDING (moving away from the truck).
- Zero (== 0):    No relative movement (matched speed or stationary relative frame).

TIME-TO-COLLISION (TTC) SPECIFICATION:
-------------------------------------------------------------------------------
TTC represents estimated time remaining until potential impact if current
trajectories and velocities persist:
- If relative_velocity_mps < 0 (approaching):
      TTC = distance_m / (-relative_velocity_mps)
- If relative_velocity_mps >= 0 (stationary or moving away):
      TTC is clamped to MAX_SAFE_TTC_S (default: 99.9 s) to represent "no immediate
      collision trajectory", avoiding division by zero, NaN, or infinite values.
"""

from __future__ import annotations

from typing import Any, Iterable
import numpy as np
import pandas as pd

# =============================================================================
# 1. CONSTANTS & DOMAIN VALUES
# =============================================================================

# Maximum time-to-collision assigned when an object is receding or stationary.
# Using a large finite number (99.9s) rather than np.inf ensures gradient-boosted
# decision trees (XGBoost, HistGradientBoosting) do not fail or produce NaN splits.
MAX_SAFE_TTC_S: float = 99.9

# Candidate object categories recognizable by mining perception / clustering models
OBJECT_TYPES: list[str] = [
    "car",        # Light utility vehicle / supervisor pickup
    "truck",      # Haul truck / dump truck / water tanker
    "crane",      # Mobile crane / lifting equipment
    "excavator",  # Shovel / excavator / backhoe
    "person",     # Ground personnel / miner / technician
    "unknown",    # Unclassified cluster / boulder / debris
]

# Standard safety risk tiers (ordered from lowest risk to emergency)
RISK_CLASSES: list[str] = [
    "SAFE",      # Normal mining operations; safe distance and clearance
    "CAUTION",   # Object detected in operational zone; heightened driver awareness
    "WARNING",   # Impending proximity or hazardous closing speed; braking required
    "CRITICAL",  # Imminent collision hazard or emergency threshold breach
]

# Ordinal mapping useful when models require numeric target encodings
RISK_CLASS_TO_INT: dict[str, int] = {
    "SAFE": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}

INT_TO_RISK_CLASS: dict[int, str] = {v: k for k, v in RISK_CLASS_TO_INT.items()}


# =============================================================================
# 2. FEATURE GROUPS
# =============================================================================

# Spatial, dimensional, and kinematic properties extracted from LiDAR/sensor clusters
OBJECT_FEATURES: list[str] = [
    "distance_m",             # Euclidean distance from truck sensor to object center (m)
    "object_x_m",             # Longitudinal position ahead of truck (+X forward) (m)
    "object_y_m",             # Lateral position relative to truck centerline (+/-Y) (m)
    "object_z_m",             # Vertical position relative to truck sensor mount (+Z up) (m)
    "object_width_m",         # Estimated 3D bounding box width (lateral dimension) (m)
    "object_height_m",        # Estimated 3D bounding box height (vertical dimension) (m)
    "object_length_m",        # Estimated 3D bounding box length (longitudinal dimension) (m)
    "point_count",            # Number of LiDAR points clustered inside bounding box
    "relative_velocity_mps",  # Relative closing velocity (m/s, negative = approaching)
    "object_type",            # Categorical object classification label
]

# Haul truck operational state (from CAN bus, GPS, or truck telemetry)
TRUCK_FEATURES: list[str] = [
    "truck_speed_kmph",       # Current ground speed of the haul truck (km/h)
]

# Dynamically derived kinematic safety metrics
DERIVED_FEATURES: list[str] = [
    "time_to_collision_s",    # Derived Time-to-Collision (seconds, clamped <= 99.9)
]

# Complete set of input features required for model training and inference
FEATURE_COLUMNS: list[str] = OBJECT_FEATURES + TRUCK_FEATURES + DERIVED_FEATURES

# Numerical input features
NUMERICAL_FEATURES: list[str] = [
    "distance_m",
    "object_x_m",
    "object_y_m",
    "object_z_m",
    "object_width_m",
    "object_height_m",
    "object_length_m",
    "point_count",
    "relative_velocity_mps",
    "truck_speed_kmph",
    "time_to_collision_s",
]

# Categorical input features (require encoding before feeding to ML models)
CATEGORICAL_FEATURES: list[str] = [
    "object_type",
]

# Target column name
TARGET_COLUMN: str = "risk_level"


# =============================================================================
# 3. KINEMATIC COMPUTATION HELPERS
# =============================================================================

def calculate_ttc(
    distance_m: float | np.ndarray | pd.Series,
    relative_velocity_mps: float | np.ndarray | pd.Series,
    max_ttc_s: float = MAX_SAFE_TTC_S,
) -> float | np.ndarray | pd.Series:
    """Calculate Time-to-Collision (TTC) safely for scalars, numpy arrays, or pandas Series.

    Kinematic Formula:
        If relative_velocity_mps < 0 (object is closing in):
            TTC = distance_m / -relative_velocity_mps
        If relative_velocity_mps >= 0 (object is receding or stationary):
            TTC = max_ttc_s (default: 99.9s)

    Guarantees:
    - Never returns NaN, inf, or negative TTC values.
    - Zero relative velocity is safely treated as no closing trajectory (max_ttc_s).

    Args:
        distance_m: Radial distance to the target object in meters (>= 0).
        relative_velocity_mps: Relative velocity in m/s (negative = closing in).
        max_ttc_s: Ceiling value representing safe / non-approaching scenarios.

    Returns:
        TTC in seconds (float, np.ndarray, or pd.Series matching input type).
    """
    # Vectorized computation for pandas Series or numpy arrays
    if isinstance(distance_m, (pd.Series, np.ndarray)) or isinstance(relative_velocity_mps, (pd.Series, np.ndarray)):
        dist = np.asarray(distance_m, dtype=float)
        rel_vel = np.asarray(relative_velocity_mps, dtype=float)

        # Boolean mask: object is approaching when relative velocity is strictly negative
        approaching_mask = rel_vel < -1e-4

        # Initialize full array with max_ttc_s
        ttc = np.full_like(dist, fill_value=max_ttc_s, dtype=float)

        # Compute TTC where approaching
        closing_speed = -rel_vel[approaching_mask]
        computed = np.where(closing_speed > 0, dist[approaching_mask] / closing_speed, max_ttc_s)

        # Clamp between 0.0 and max_ttc_s
        ttc[approaching_mask] = np.clip(computed, 0.0, max_ttc_s)

        if isinstance(distance_m, pd.Series):
            return pd.Series(ttc, index=distance_m.index, name="time_to_collision_s")
        return ttc

    # Scalar float computation
    d = float(distance_m)
    v = float(relative_velocity_mps)

    if v < -1e-4:
        closing_speed = -v
        computed_ttc = d / closing_speed
        return float(min(max(computed_ttc, 0.0), max_ttc_s))

    return float(max_ttc_s)


# =============================================================================
# 4. DATA VALIDATION HELPERS
# =============================================================================

def validate_schema(
    df: pd.DataFrame,
    require_target: bool = True,
) -> tuple[bool, list[str]]:
    """Validate that a DataFrame conforms to the mineRakshak feature schema.

    Checks:
    1. All expected feature columns are present.
    2. If require_target is True, the target column 'risk_level' is present.
    3. Categorical values in 'object_type' belong to known OBJECT_TYPES.
    4. Target values belong to known RISK_CLASSES (if target is present).
    5. No NaN or infinite values exist in numerical columns.

    Args:
        df: Input DataFrame to check.
        require_target: If True, checks for presence and validity of target column.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    # 1. Check required feature columns
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {missing_features}")

    # 2. Check target column
    if require_target:
        if TARGET_COLUMN not in df.columns:
            errors.append(f"Missing target column: '{TARGET_COLUMN}'")
        else:
            invalid_targets = set(df[TARGET_COLUMN].dropna().unique()) - set(RISK_CLASSES)
            if invalid_targets:
                errors.append(f"Invalid target classes found in '{TARGET_COLUMN}': {invalid_targets}")

    # 3. Check categorical values
    if "object_type" in df.columns:
        invalid_types = set(df["object_type"].dropna().unique()) - set(OBJECT_TYPES)
        if invalid_types:
            errors.append(f"Invalid categories found in 'object_type': {invalid_types}")

    # 4. Check for NaNs or Infs in numerical features
    present_numerical = [c for c in NUMERICAL_FEATURES if c in df.columns]
    for col in present_numerical:
        num_nans = df[col].isna().sum()
        if num_nans > 0:
            errors.append(f"Column '{col}' has {num_nans} NaN value(s).")
        # Check for infinities
        num_infs = np.isinf(df[col]).sum()
        if num_infs > 0:
            errors.append(f"Column '{col}' has {num_infs} Infinite value(s).")

    is_valid = len(errors) == 0
    return is_valid, errors
