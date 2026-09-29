"""Synthetic Dataset Generator for mineRakshak-ai ML Risk Model.

===============================================================================
IMPORTANT DISCLAIMER:
===============================================================================
This script produces SYNTHETIC sensor-derived features representing what a future
3D LiDAR point-cloud processing and clustering pipeline is expected to produce.
We currently do NOT have physical access to live ROS 2 LiDAR data streams.
This dataset is strictly a temporary engineering proxy for developing the ML
risk classification pipeline and MUST NOT be treated as real-world mine telemetry.
===============================================================================

Physics & Perception Simulation Highlights:
1. Multi-modal object dimensions conditioned on object_type (cars, haul trucks,
   excavators, cranes, personnel, unknown debris).
2. Geometrically consistent 3D positions (X forward, Y lateral, Z vertical) where
   distance_m = sqrt(X^2 + Y^2 + Z^2) with realistic LiDAR measurement noise.
3. LiDAR point count simulated via physical cross-sectional area and inverse-square
   range attenuation with reflectivity noise.
4. Haul truck operational ground speed distributions (stopped, pit crawl, haulway).
5. Kinematic closing velocity (negative = approaching, positive = receding).
6. Multi-factor non-linear risk labeling incorporating distance, time-to-collision,
   closing speed, truck speed, lateral travel corridor relevance, and object type,
   with realistic noise creating non-trivial decision boundary overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so schema can be imported cleanly
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from src.preprocessing.schema import (
    FEATURE_COLUMNS,
    OBJECT_TYPES,
    RISK_CLASSES,
    TARGET_COLUMN,
    calculate_ttc,
    validate_schema,
)

# Output dataset path
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "synthetic_mine_data.csv"

# Configuration constants
DEFAULT_NUM_SAMPLES = 20_000
RANDOM_SEED = 42

# Plausible 3D bounding box dimension distributions per object category
# Format: (mean_w, std_w, min_w, max_w), (mean_h, std_h, min_h, max_h), (mean_l, std_l, min_l, max_l)
OBJECT_DIMENSION_SPECS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "person": {
        "width":  (0.55, 0.08, 0.40, 0.80),
        "height": (1.72, 0.12, 1.45, 2.05),
        "length": (0.45, 0.08, 0.30, 0.70),
    },
    "car": {
        "width":  (1.90, 0.15, 1.60, 2.30),
        "height": (1.60, 0.15, 1.35, 2.05),
        "length": (4.60, 0.35, 3.80, 5.60),
    },
    "truck": {
        "width":  (4.20, 0.60, 3.00, 6.50),
        "height": (4.50, 0.60, 3.20, 6.80),
        "length": (9.50, 1.20, 6.50, 13.50),
    },
    "excavator": {
        "width":  (4.50, 0.50, 3.50, 6.20),
        "height": (4.80, 0.60, 3.50, 6.50),
        "length": (9.80, 1.10, 7.00, 13.00),
    },
    "crane": {
        "width":  (3.60, 0.40, 2.80, 4.80),
        "height": (6.20, 0.80, 4.20, 9.00),
        "length": (10.50, 1.20, 7.50, 14.50),
    },
    "unknown": {
        "width":  (1.60, 0.60, 0.40, 3.50),
        "height": (1.30, 0.50, 0.30, 3.20),
        "length": (1.80, 0.70, 0.40, 4.50),
    },
}

# Object type frequency in typical surface mining scenarios
OBJECT_TYPE_WEIGHTS = {
    "truck": 0.28,      # Other haul trucks, water tankers
    "car": 0.24,        # Utility pickups, inspection SUVs
    "excavator": 0.16,  # Digging & loading equipment
    "crane": 0.10,      # Maintenance & lifting cranes
    "person": 0.11,     # Ground miners, spotters, surveyors
    "unknown": 0.11,    # Boulders, tire barriers, dust clusters
}


def sample_dimensions(obj_type: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample realistic width, height, length bounded by physical object constraints."""
    spec = OBJECT_DIMENSION_SPECS[obj_type]

    def _sample(stat: tuple[float, float, float, float]) -> np.ndarray:
        mean, std, min_v, max_v = stat
        vals = rng.normal(mean, std, size=n)
        return np.clip(vals, min_v, max_v)

    w = _sample(spec["width"])
    h = _sample(spec["height"])
    l = _sample(spec["length"])
    return w, h, l


def calculate_risk_label(
    distance: np.ndarray,
    ttc: np.ndarray,
    rel_vel: np.ndarray,
    truck_speed: np.ndarray,
    obj_x: np.ndarray,
    obj_y: np.ndarray,
    obj_type: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multi-factor non-linear risk assessment with realistic operational boundary noise.

    Combines:
    1. Distance score (closer = higher risk)
    2. TTC score (imminent impact < 4s = drastic risk spike)
    3. Closing speed penalty (approaching fast = severe hazard)
    4. Lateral corridor relevance (|y| within truck travel path ~2.5m = direct danger;
       wide lateral offset = mitigated danger)
    5. Truck speed factor (higher speed increases reaction distance)
    6. Vulnerability factor (pedestrians near truck elevate risk)
    7. Noise perturbation (simulates unmodeled road grade, dust, operator uncertainty)
    """
    n = len(distance)
    risk_score = np.zeros(n, dtype=float)

    # 1. Lateral corridor relevance factor
    # Haul truck width is ~4.5m, so travel corridor is roughly |y| <= 2.5m
    abs_y = np.abs(obj_y)
    path_factor = np.clip(1.0 - (abs_y / 8.0), 0.15, 1.0)
    # At close range (< 7m), blind spot hazards are elevated even if slightly off-center
    path_factor[distance < 7.0] = np.maximum(path_factor[distance < 7.0], 0.85)

    # 2. Proximity score (0 to 35 pts)
    # Higher for in-path objects, but close objects always contribute baseline points
    dist_pts = 35.0 * np.clip(1.0 - (distance / 55.0), 0.0, 1.0)
    risk_score += dist_pts * (0.35 + 0.65 * path_factor)

    # 3. Time-to-Collision (TTC) score (0 to 45 pts)
    # For approaching objects with finite TTC (< 7.0s)
    approaching = rel_vel < -0.1
    ttc_pts = np.where(approaching, 45.0 * np.clip(1.0 - (ttc / 7.0), 0.0, 1.0) * path_factor, 0.0)
    risk_score += ttc_pts

    # 4. Closing speed & truck speed dynamic hazard bonus (0 to 22 pts)
    closing_speed = np.maximum(-rel_vel, 0.0)
    speed_pts = (closing_speed / 18.0) * 10.0 * path_factor + (truck_speed / 55.0) * 12.0 * path_factor
    risk_score += speed_pts

    # 5. Stationary obstacle in direct path:
    # If truck is moving fast (>15 km/h) and stationary object is close ahead (<25m) in path
    stationary_in_path = (np.abs(rel_vel) < 0.5) & (distance < 25.0) & (truck_speed > 15.0) & (abs_y <= 2.8)
    risk_score[stationary_in_path] += 18.0

    # 6. Vulnerable road user (person) special safety protection bonus
    # A person within 25m of haul truck demands heightened driver caution
    is_person = obj_type == "person"
    person_close = is_person & (distance < 25.0) & (abs_y < 7.0)
    risk_score[person_close] += 14.0

    # 7. Receding object relief (moving away fast reduces risk)
    receding_fast = rel_vel > 1.5
    risk_score[receding_fast] *= 0.65

    # 8. Realistic stochastic perturbation (simulates unmodeled road roughness, sensor jitter, noise)
    noise = rng.normal(0.0, 3.5, size=n)
    final_score = risk_score + noise

    # Balanced risk thresholds calibrated to target distributions:
    # SAFE: 35-45%, CAUTION: 20-30%, WARNING: 15-25%, CRITICAL: 10-20%
    labels = np.empty(n, dtype=object)
    labels[final_score < 14.0] = "SAFE"
    labels[(final_score >= 14.0) & (final_score < 26.0)] = "CAUTION"
    labels[(final_score >= 26.0) & (final_score < 47.0)] = "WARNING"
    labels[final_score >= 47.0] = "CRITICAL"

    return labels


def generate_synthetic_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a high-fidelity synthetic tabular dataset conforming to the mineRakshak schema."""
    rng = np.random.Generator(np.random.PCG64(seed))

    # 1. Sample object types according to operational weights
    types_list = list(OBJECT_TYPE_WEIGHTS.keys())
    weights = [OBJECT_TYPE_WEIGHTS[t] for t in types_list]
    obj_types = rng.choice(types_list, size=num_samples, p=weights)

    # 2. Sample 3D physical dimensions per category
    widths = np.zeros(num_samples, dtype=float)
    heights = np.zeros(num_samples, dtype=float)
    lengths = np.zeros(num_samples, dtype=float)

    for o_type in types_list:
        mask = obj_types == o_type
        count = np.sum(mask)
        if count > 0:
            w, h, l = sample_dimensions(o_type, count, rng)
            widths[mask] = w
            heights[mask] = h
            lengths[mask] = l

    # 3. Sample truck operational speed (km/h)
    # Haul trucks operate 0 to 55 km/h across distinct operational modes:
    # - Stopped/shoveling/queueing (0-5 km/h, 15%)
    # - Slow pit navigation / loading (5-25 km/h, 35%)
    # - Haul road transport (25-45 km/h, 35%)
    # - Higher speed open transport (45-55 km/h, 15%)
    mode = rng.choice([0, 1, 2, 3], size=num_samples, p=[0.15, 0.35, 0.35, 0.15])
    truck_speed_kmph = np.zeros(num_samples, dtype=float)
    truck_speed_kmph[mode == 0] = rng.uniform(0.0, 5.0, size=np.sum(mode == 0))
    truck_speed_kmph[mode == 1] = rng.uniform(5.0, 25.0, size=np.sum(mode == 1))
    truck_speed_kmph[mode == 2] = rng.uniform(25.0, 45.0, size=np.sum(mode == 2))
    truck_speed_kmph[mode == 3] = rng.uniform(45.0, 55.0, size=np.sum(mode == 3))
    truck_speed_kmph = np.round(truck_speed_kmph, 2)
    truck_speed_mps = truck_speed_kmph / 3.6

    # 4. Sample spatial positions (X forward, Y lateral, Z vertical)
    # Forward distance X: close proximity (2.5-22m, 42%), medium (22-50m, 38%), distant (50-100m, 20%)
    dist_zone = rng.choice([0, 1, 2], size=num_samples, p=[0.42, 0.38, 0.20])
    obj_x = np.zeros(num_samples, dtype=float)
    obj_x[dist_zone == 0] = rng.uniform(2.5, 22.0, size=np.sum(dist_zone == 0))
    obj_x[dist_zone == 1] = rng.uniform(22.0, 50.0, size=np.sum(dist_zone == 1))
    obj_x[dist_zone == 2] = rng.uniform(50.0, 100.0, size=np.sum(dist_zone == 2))

    # Lateral position Y: centered on 0, with spread matching travel corridor and adjacent lanes
    # 48% in direct travel corridor (|Y| <= 2.5m), 34% adjacent/shoulder, 18% wider berm
    y_zone = rng.choice([0, 1, 2], size=num_samples, p=[0.48, 0.34, 0.18])
    obj_y = np.zeros(num_samples, dtype=float)
    obj_y[y_zone == 0] = rng.normal(0.0, 1.3, size=np.sum(y_zone == 0))
    obj_y[y_zone == 1] = rng.choice([-1, 1], size=np.sum(y_zone == 1)) * rng.uniform(2.5, 6.0, size=np.sum(y_zone == 1))
    obj_y[y_zone == 2] = rng.choice([-1, 1], size=np.sum(y_zone == 2)) * rng.uniform(6.0, 16.0, size=np.sum(y_zone == 2))

    # Vertical position Z: relative to truck LiDAR sensor origin (assumed mounted at ~2.5m height)
    # Centroid Z depends on object height plus ground elevation variations
    sensor_height = 2.5
    obj_z = (heights / 2.0) - sensor_height + rng.normal(0.0, 0.25, size=num_samples)

    # 5. Geometrically consistent distance with slight LiDAR measurement noise
    true_distance = np.sqrt(obj_x**2 + obj_y**2 + obj_z**2)
    distance_noise = rng.normal(0.0, 0.05, size=num_samples)
    distance_m = np.maximum(true_distance + distance_noise, 1.0)

    # 6. Relative velocity (m/s, negative = closing, positive = receding)
    # Mixture of:
    # a) Approaching oncoming traffic or closing in on stationary obstacles: v_rel < 0 (~60%)
    # b) Matched speed / stationary relative: v_rel approx 0 (~14%)
    # c) Receding vehicles / moving away: v_rel > 0 (~26%)
    vel_scenario = rng.choice([0, 1, 2], size=num_samples, p=[0.60, 0.14, 0.26])
    rel_vel = np.zeros(num_samples, dtype=float)

    # Approaching (closing in)
    n_app = np.sum(vel_scenario == 0)
    app_types = rng.choice([0, 1, 2], size=n_app, p=[0.45, 0.25, 0.30])
    v_app = np.zeros(n_app, dtype=float)
    # Stationary obstacle approached by truck
    v_app[app_types == 0] = -truck_speed_mps[vel_scenario == 0][app_types == 0] + rng.normal(0.0, 0.3, size=np.sum(app_types == 0))
    # Oncoming traffic
    v_oncoming_other = rng.uniform(4.0, 14.0, size=np.sum(app_types == 1))
    v_app[app_types == 1] = -(truck_speed_mps[vel_scenario == 0][app_types == 1] + v_oncoming_other)
    # Slower lead vehicle
    lead_diff = rng.uniform(1.0, 6.0, size=np.sum(app_types == 2))
    v_app[app_types == 2] = -np.minimum(lead_diff, truck_speed_mps[vel_scenario == 0][app_types == 2])
    rel_vel[vel_scenario == 0] = np.clip(v_app, -25.0, -0.2)

    # Stationary relative
    n_stat = np.sum(vel_scenario == 1)
    rel_vel[vel_scenario == 1] = rng.normal(0.0, 0.15, size=n_stat)

    # Receding
    n_rec = np.sum(vel_scenario == 2)
    rel_vel[vel_scenario == 2] = rng.uniform(0.3, 14.0, size=n_rec)

    # 7. LiDAR point count simulation
    # Point density scales with cross-sectional area and attenuates with distance
    cross_section = widths * heights
    base_points = 18_000.0 * (cross_section / (distance_m + 1.5)**1.75)
    reflectivity_factor = rng.lognormal(mean=0.0, sigma=0.30, size=num_samples)
    point_count = np.clip(np.round(base_points * reflectivity_factor), 2, 8500).astype(int)

    # 8. Derived Time-to-Collision (TTC) using official project schema logic
    ttc_s = calculate_ttc(distance_m, rel_vel)

    # 9. Multi-factor non-linear Risk Label assignment
    risk_labels = calculate_risk_label(
        distance=distance_m,
        ttc=ttc_s,
        rel_vel=rel_vel,
        truck_speed=truck_speed_kmph,
        obj_x=obj_x,
        obj_y=obj_y,
        obj_type=obj_types,
        rng=rng,
    )

    # Assemble final DataFrame strictly matching FEATURE_COLUMNS + [TARGET_COLUMN]
    df = pd.DataFrame({
        "distance_m": np.round(distance_m, 2),
        "object_x_m": np.round(obj_x, 2),
        "object_y_m": np.round(obj_y, 2),
        "object_z_m": np.round(obj_z, 2),
        "object_width_m": np.round(widths, 2),
        "object_height_m": np.round(heights, 2),
        "object_length_m": np.round(lengths, 2),
        "point_count": point_count,
        "relative_velocity_mps": np.round(rel_vel, 2),
        "object_type": obj_types,
        "truck_speed_kmph": truck_speed_kmph,
        "time_to_collision_s": np.round(ttc_s, 2),
        "risk_level": risk_labels,
    })

    return df


def validate_generated_data(df: pd.DataFrame, expected_rows: int = DEFAULT_NUM_SAMPLES) -> None:
    """Rigorous post-generation integrity and physical sanity validation."""
    print("Validating generated synthetic dataset against schema...", flush=True)

    # 1. Schema check
    is_valid, errors = validate_schema(df, require_target=True)
    if not is_valid:
        raise ValueError(f"Schema validation failed with errors: {errors}")

    # 2. Row count check
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, but got {len(df)}.")

    # 3. Column check
    expected_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Columns do not match expected order.\nGot: {list(df.columns)}\nExpected: {expected_cols}")

    # 4. Physical value sanity bounds
    assert (df["distance_m"] >= 0).all(), "Negative distance found!"
    assert (df["truck_speed_kmph"] >= 0).all(), "Negative truck speed found!"
    assert (df["object_width_m"] > 0).all(), "Zero or negative width found!"
    assert (df["object_height_m"] > 0).all(), "Zero or negative height found!"
    assert (df["object_length_m"] > 0).all(), "Zero or negative length found!"
    assert (df["point_count"] > 0).all(), "Zero or negative point count found!"
    assert (df["time_to_collision_s"] >= 0).all(), "Negative TTC found!"
    assert not np.isinf(df["time_to_collision_s"]).any(), "Infinite TTC found!"

    # 5. Geometrical consistency check: distance should approximate sqrt(x^2 + y^2 + z^2) within noise
    computed_dist = np.sqrt(df["object_x_m"]**2 + df["object_y_m"]**2 + df["object_z_m"]**2)
    max_geom_diff = (df["distance_m"] - computed_dist).abs().max()
    assert max_geom_diff < 0.5, f"Geometrical position inconsistency detected! Max diff = {max_geom_diff:.3f}m"

    print("All physical, geometrical, and schema validation checks PASSED successfully!\n", flush=True)


def main() -> None:
    print("=" * 70, flush=True)
    print("mineRakshak-ai: Generating Synthetic Sensor & Object Dataset", flush=True)
    print(f"Target sample count: {DEFAULT_NUM_SAMPLES:,} rows", flush=True)
    print(f"Random seed: {RANDOM_SEED}", flush=True)
    print("=" * 70, flush=True)

    df = generate_synthetic_dataset(num_samples=DEFAULT_NUM_SAMPLES, seed=RANDOM_SEED)

    # Validate dataset before saving
    validate_generated_data(df, expected_rows=DEFAULT_NUM_SAMPLES)

    # Ensure output directory exists
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset successfully saved to: {RAW_DATA_PATH}\n", flush=True)

    print("Dataset generated successfully\n", flush=True)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Feature List: {list(df.columns)}\n")

    print("Risk distribution:")
    counts = df["risk_level"].value_counts()
    percentages = df["risk_level"].value_counts(normalize=True) * 100
    for r_class in RISK_CLASSES:
        c = counts.get(r_class, 0)
        p = percentages.get(r_class, 0.0)
        print(f"  {r_class}: {c:,} ({p:.1f}%)")

    print(f"\nDistance range: {df['distance_m'].min():.2f} m to {df['distance_m'].max():.2f} m")
    print(f"Truck speed range: {df['truck_speed_kmph'].min():.2f} km/h to {df['truck_speed_kmph'].max():.2f} km/h")
    print(f"TTC range: {df['time_to_collision_s'].min():.2f} s to {df['time_to_collision_s'].max():.2f} s")

    print("\nFirst 8 rows of generated dataset:")
    print(df.head(8).to_string())


if __name__ == "__main__":
    main()
